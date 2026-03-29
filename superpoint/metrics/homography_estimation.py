import cv2
import keras
import tensorflow as tf
from constants import SP_INPUT_SHAPE



class HomographyEstimationMetric(keras.metrics.Metric):
    """
    Homography Estimation metric from SuperPoint paper.
    
    Measures the ability of an algorithm to estimate the homography relating
    a pair of images by comparing how well the homography transforms the four
    corners of one image onto the other.
    
    Args:
        pixel_threshold: Distance threshold (ε) for considering a corner transformation correct (in pixels)
        correctness_threshold: Threshold (η) for mean corner accuracy to consider homography correct
        name: Name of the metric
        dtype: Data type for computations
    """
    
    def __init__(self, 
                 correctness_threshold: float = 3.0,
                 name: str = 'homography_estimation', 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.correctness_threshold = correctness_threshold  # η threshold
        
        # State variables
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')
    
    def update_state(self, 
                     keypoints1: tf.Tensor,
                     descriptors1: tf.Tensor, 
                     keypoints2: tf.Tensor,
                     descriptors2: tf.Tensor,
                     ground_truth_homography: tf.Tensor,
                    ):
        """
        Update metric state with a batch of predictions.
        
        Args:
            keypoints1: Keypoints from first image [N, 2]
            descriptors1: Descriptors from first image [N, D]
            keypoints2: Keypoints from second image [M, 2]
            descriptors2: Descriptors from second image [M, D]
            ground_truth_homography: Ground truth homography matrix [3, 3]
        """
        
        
        src_pts, dst_pts = self._src_dst_points(
            keypoints1, descriptors1, keypoints2, descriptors2
        )
        # Estimate homography using nearest neighbor matching + RANSAC
        estimated_homography = self._estimate_homography(src_pts, dst_pts)
        
        if estimated_homography is not None:
            # Evaluate homography quality using corner comparison
            corner_accuracy = self._evaluate_homography(
                estimated_homography, ground_truth_homography
            )
            
            self.total_correct.assign_add(corner_accuracy)
            self.total_count.assign_add(1.0)
            
            
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 256], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 256], dtype=tf.float32)
        ),
        jit_compile=True
    )
    def _src_dst_points(self, 
                        keypoints1: tf.Tensor,
                        descriptors1: tf.Tensor,
                        keypoints2: tf.Tensor, 
                        descriptors2: tf.Tensor
                        ): 
        dist = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(descriptors1, 1) - descriptors2[tf.newaxis, ...]), axis=-1))
        sorted_indices = tf.argsort(dist, axis=-1)[:, 0]
        sorted_dist = tf.gather(dist, sorted_indices, batch_dims=1)
        
        src_pts = tf.expand_dims(keypoints1, axis=1)
        dst_pts = tf.expand_dims(tf.gather(keypoints2, sorted_indices), axis=1)
        
        mask = sorted_dist < 6.3
        src_pts = tf.boolean_mask(src_pts, mask, axis=0)
        dst_pts = tf.boolean_mask(dst_pts, mask, axis=0)
        
        return src_pts, dst_pts
               
    
    def _estimate_homography(self, 
                           src_pts: tf.Tensor,
                           dst_pts: tf.Tensor
                           ):
        """
        Estimate homography using nearest neighbor matching and RANSAC.
        
        Returns:
            Estimated homography matrix [3, 3] or None if estimation fails
        """
        
        if len(src_pts) < 0 or len(dst_pts) < 4:
            return None
        
        # Estimate homography using RANSAC
        try:
            homography, mask = cv2.findHomography(
                src_pts.numpy(), dst_pts.numpy(), 
                cv2.RANSAC, 
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )
            return homography
        except:
            return None
    
    
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[3, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[3, 3], dtype=tf.float32)
        ),
        jit_compile=True
    )
    def _evaluate_homography(self, estimated_H, ground_truth_H):
        """
        Evaluate homography by comparing corner transformations.
        
        Following the paper's formula:
        CorrH = 1/N Σ(i=1 to N) [1/4 Σ(j=1 to 4) 1{||c'ij - ĉ'ij|| < ε}]
        
        Args:
            estimated_H: Estimated homography matrix [3, 3]
            ground_truth_H: Ground truth homography matrix [3, 3]
            image_shape: Image shape (height, width)
            
        Returns:
            1.0 if homography is correct (mean corner accuracy >= η), 0.0 otherwise
        """
        
        height, width = SP_INPUT_SHAPE[:-1]
        
        # Define four corners of the first image
        corners = tf.constant([
            [0, 0],           # Top-left
            [width-1, 0],     # Top-right
            [width-1, height-1], # Bottom-right
            [0, height-1]     # Bottom-left
        ], dtype=tf.float32)
        
        # Transform corners using ground truth homography
        corners_gt = self._transform_points(corners, ground_truth_H)
        
        # Transform corners using estimated homography
        corners_est = self._transform_points(corners, estimated_H)
        
        # Compute distances between corresponding corners
        distances = tf.linalg.norm(corners_gt - corners_est, axis=1)
        
        # Take mean of distances
        mean_corner_distance = tf.reduce_mean(distances)
        
        # Homography is considered correct if mean distance <= e (correctness_threshold)
        return tf.cast(mean_corner_distance <= self.correctness_threshold, tf.float32)
    
  
    def _transform_points(self, 
                          points, 
                          homography
                          ):
        """
        Transform points using homography matrix.
        
        Args:
            points: Points to transform [N, 2]
            homography: Homography matrix [3, 3]
            
        Returns:
            Transformed points [N, 2]
        """
        
        # Convert to homogeneous coordinates
        points_h = tf.concat([points, tf.ones(shape=(tf.shape(points)[0], 1))], axis=-1)
        
        # Apply homography transformation
        transformed_h = tf.matmul(homography, tf.transpose(points_h))
        
        # Convert back to Cartesian coordinates
        transformed = tf.transpose(transformed_h[:2] / transformed_h[2])
        
        return transformed
    
    def result(self) -> tf.Tensor:
        """
        Compute the final metric result.
        
        Returns:
            Homography estimation accuracy (CorrH score from paper)
        """
        return tf.math.divide_no_nan(self.total_correct, self.total_count)
    
    def reset_state(self):
        """Reset the metric state."""
        self.total_correct.assign(0.0)
        self.total_count.assign(0.0)
