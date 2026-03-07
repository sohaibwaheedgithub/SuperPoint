import keras
import tensorflow as tf



class Repeatability(keras.metrics.Metric):
    """
    Repeatability metric implementation for SuperPoint as described in:
    "SuperPoint: Self-Supervised Interest Point Detection and Description"
    
    This metric measures the probability that a point detected in one image
    is also detected in a second image (after geometric transformation).
    """
    
    def __init__(self, distance_threshold=3.0, name='repeatability', **kwargs):
        """
        Args:
            distance_threshold (float): Distance threshold ε for considering 
                                      two points as corresponding
            name (str): Name of the metric
        """
        super(Repeatability, self).__init__(name=name, **kwargs)
        self.distance_threshold = distance_threshold
        
        # Initialize state variables
        self.total_repeatability = self.add_weight(
            name='total_repeatability', initializer='zeros', dtype=tf.float32
        )
        self.count = self.add_weight(
            name='count', initializer='zeros', dtype=tf.float32
        )
    
    @tf.function(
            input_signature=(
                tf.TensorSpec([None, 2], tf.float32),
                tf.TensorSpec([None, 2], tf.float32),
                tf.TensorSpec([3, 3], tf.float32)
            ),
            jit_compile=True

    )
    def update_state(self, points1, points2, homography):
        """
        Update the metric state.
        
        Args:
            - 'points1': Detected points in first image [N1, 2]
            - 'points2': Detected points in second image [N2, 2] 
            - 'homography': Homography matrix [batch_size, 3, 3]
        """
        
        # Transform points1 using homography
        pts1_transformed = self._transform_points(points1, homography)
        
        # Compute repeatability for this pair
        rep_score = self._compute_repeatability(pts1_transformed, points2)
        
        self.total_repeatability.assign_add(rep_score)
        self.count.assign_add(1.)
        
    def _transform_points(self, points, homography):
        """
        Transform 2D points using homography matrix.
        
        Args:
            points: Points to transform [N, 2]
            homography: Homography matrix [3, 3]
            
        Returns:
            Transformed points [N, 2]
        """
        # Convert to homogeneous coordinates
        ones = tf.ones([tf.shape(points)[0], 1], dtype=points.dtype)
        points_homo = tf.concat([points, ones], axis=1)  # [N, 3]
        
        # Apply homography transformation
        points_transformed_homo = tf.matmul(points_homo, homography, transpose_b=True)  # [N, 3]
        
        # Convert back to 2D coordinates (normalize by z)
        points_transformed = points_transformed_homo[:, :2] / tf.expand_dims(
            points_transformed_homo[:, 2], axis=1
        )
        
        return points_transformed
    
    def _compute_repeatability(self, points1_transformed, points2):
        """
        Compute repeatability between two sets of points.
        
        Args:
            points1_transformed: Transformed points from first image [N1, 2]
            points2: Points from second image [N2, 2]
            
        Returns:
            Repeatability score (scalar)
        """
        N1 = tf.shape(points1_transformed)[0]
        N2 = tf.shape(points2)[0]
        
        # Handle empty point sets
        if N1 == 0 and N2 == 0:
            return tf.constant(1.0, dtype=tf.float32)  # Perfect repeatability for empty sets
        elif N1 == 0 or N2 == 0:
            return tf.constant(0.0, dtype=tf.float32)  # No repeatability if one set is empty
        else:
            # Compute correctness for points1 -> points2
            corr1 = self._compute_correctness(points1_transformed, points2)
            
            # Compute correctness for points2 -> points1 (reverse direction)
            corr2 = self._compute_correctness(points2, points1_transformed)
            
            
            # Compute repeatability as per equation (14)
            total_correct = tf.reduce_sum(corr1) + tf.reduce_sum(corr2)
            total_points = tf.cast(N1 + N2, tf.float32)
            
            repeatability = total_correct / total_points
            
            return repeatability
    
    def _compute_correctness(self, points_src, points_dst):
        """
        Compute correctness for each point in points_src.
        
        Args:
            points_src: Source points [N_src, 2]
            points_dst: Destination points [N_dst, 2]
            
        Returns:
            Correctness for each source point [N_src]
        """
        # Compute pairwise distances
        # points_src: [N_src, 2], points_dst: [N_dst, 2]
        points_src_expanded = tf.expand_dims(points_src, axis=1)  # [N_src, 1, 2]
        points_dst_expanded = tf.expand_dims(points_dst, axis=0)  # [1, N_dst, 2]
        
        # Compute squared distances
        diff = points_src_expanded - points_dst_expanded  # [N_src, N_dst, 2]
        distances_squared = tf.reduce_sum(tf.square(diff), axis=2)  # [N_src, N_dst]istances_squared)  # [N_src, N_dst]
        
        # Find minimum distance for each source point
        min_distances = tf.reduce_min(distances_squared, axis=1)  # [N_src]
        
        # Correctness: 1 if min distance <= threshold, 0 otherwise
        correctness = tf.cast(
            min_distances <= self.distance_threshold, 
            dtype=tf.float32
        )
        
        return correctness
    
    def result(self):
        """Return the current metric result."""
        return tf.math.divide_no_nan(self.total_repeatability, self.count)
    
    def reset_state(self):
        """Reset all metric state variables."""
        self.total_repeatability.assign(0.0)
        self.count.assign(0.0)
