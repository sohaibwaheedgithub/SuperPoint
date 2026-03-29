import keras
import tensorflow as tf
from constants import SP_INPUT_SHAPE


class MatchingScore(keras.metrics.Metric):
    """
    Matching Score metric implementation for SuperPoint pipeline evaluation.
    
    This metric measures the overall performance of interest point detector 
    and descriptor combined by computing the ratio of ground truth correspondences 
    that can be recovered by the whole pipeline over the number of features 
    proposed by the pipeline in the shared viewpoint region.
    """
    
    def __init__(self, 
                 geometric_threshold=3.0,
                 descriptor_threshold=3.0,
                 name='matching_score',
                 **kwargs):
        """
        Args:
            geometric_threshold (float): Distance threshold for geometric matching (pixels)
            descriptor_threshold (float): Distance threshold for descriptor matching
            name (str): Name of the metric
        """
        super(MatchingScore, self).__init__(name=name, **kwargs)
        
        self.geometric_threshold = geometric_threshold
        self.descriptor_threshold = descriptor_threshold
        
        # Initialize state variables
        self.total_score = self.add_weight(
            name='total_score', initializer='zeros', dtype=tf.float32
        )
        self.count = self.add_weight(
            name='count', initializer='zeros', dtype=tf.float32
        )
    
    @tf.function(
        input_signature=(
            tf.TensorSpec([None, 2], tf.float32),
            tf.TensorSpec([None, 256], tf.float32),
            tf.TensorSpec([None, 2], tf.float32),
            tf.TensorSpec([None, 256], tf.float32),
            tf.TensorSpec([3, 3], tf.float32)
        ),
        jit_compile=False
    )
    def update_state(self, 
                    keypoints1: tf.Tensor,
                    descriptors1: tf.Tensor, 
                    keypoints2: tf.Tensor,
                    descriptors2: tf.Tensor,
                    homography: tf.Tensor,
                    ):
        """
        Update the metric state.
        
        Args:
            - 'keypoints1': Detected keypoints in first image [batch_size, N1, 2]
            - 'descriptors1': Descriptors for keypoints1 [batch_size, N1, 256]
            - 'keypoints2': Detected keypoints in second image [batch_size, N2, 2] 
            - 'descriptors2': Descriptors for keypoints2 [batch_size, N2, 256]
            - 'homography': Homography matrix [batch_size, 3, 3]
        """
        
            
        # Compute matching score for this pair
        score = self._compute_matching_score(keypoints1, descriptors1, keypoints2, descriptors2, homography)
            
        self.total_score.assign_add(score)
        self.count.assign_add(1.)
    
    def _compute_matching_score(self, kpts1, desc1, kpts2, desc2, homography):
        """
        Compute matching score for a pair of images.
        
        Args:
            kpts1: Keypoints from first image [N1, 2]
            desc1: Descriptors from first image [N1, 256]
            kpts2: Keypoints from second image [N2, 2]
            desc2: Descriptors from second image [N2, 256]
            homography: Homography matrix [3, 3]
            
        Returns:
            Matching score (scalar)
        """
        # Filter keypoints to shared viewpoint region
        kpts1_filtered, desc1_filtered, valid_mask1 = self._filter_to_shared_region(
            kpts1, desc1, homography
        )
        #tf.print(kpts1_filtered, desc1_filtered)
        kpts2_filtered, desc2_filtered, valid_mask2 = self._filter_to_shared_region(
            kpts2, desc2, tf.linalg.inv(homography)
        )
        #tf.print(kpts2_filtered, desc2_filtered)
        
        # Handle empty keypoint sets
        N1 = tf.shape(kpts1_filtered)[0]
        N2 = tf.shape(kpts2_filtered)[0]
        
        if N1 == 0 or N2 == 0:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Compute matching score in both directions
        score_12 = self._compute_directional_matching_score(
            kpts1_filtered, desc1_filtered, kpts2_filtered, desc2_filtered, homography
        )
        score_21 = self._compute_directional_matching_score(
            kpts2_filtered, desc2_filtered, kpts1_filtered, desc1_filtered, 
            tf.linalg.inv(homography)
        )
        
        # Average both directions
        return (score_12 + score_21) / 2.0
    
    def _filter_to_shared_region(self, keypoints, descriptors, homography):
        """
        Filter keypoints to shared viewpoint region.
        
        Args:
            keypoints: Keypoints to filter [N, 2]
            descriptors: Corresponding descriptors [N, 256]
            homography: Homography matrix [3, 3]
            
        Returns:
            filtered_keypoints: [N_filtered, 2]
            filtered_descriptors: [N_filtered, 256]
            valid_mask: [N] boolean mask
        """
        # Transform keypoints to other image coordinate system
        transformed_kpts = self._transform_points(keypoints, homography)
        
        # Check which transformed points are within image bounds
        
        valid_x = tf.logical_and(
            transformed_kpts[:, 0] >= 0,
            transformed_kpts[:, 0] < tf.cast(SP_INPUT_SHAPE[1], tf.float32)
        )
        valid_y = tf.logical_and(
            transformed_kpts[:, 1] >= 0,
            transformed_kpts[:, 1] < tf.cast(SP_INPUT_SHAPE[0], tf.float32)
        )
        
        valid_mask = tf.logical_and(valid_x, valid_y)
        
        # Filter keypoints and descriptors
        filtered_keypoints = tf.boolean_mask(keypoints, valid_mask)
        filtered_descriptors = tf.boolean_mask(descriptors, valid_mask)
        
        return filtered_keypoints, filtered_descriptors, valid_mask
    
    def _compute_directional_matching_score(self, kpts_query, desc_query, kpts_db, desc_db, homography):
        """
        Compute matching score in one direction.
        
        Args:
            kpts_query: Query keypoints [N_query, 2]
            desc_query: Query descriptors [N_query, 256]
            kpts_db: Database keypoints [N_db, 2]
            desc_db: Database descriptors [N_db, 256]
            homography: Homography matrix [3, 3]
            
        Returns:
            Matching score (scalar)
        """
        # L2 normalize descriptors
        desc_query_norm = tf.nn.l2_normalize(desc_query, axis=1)
        desc_db_norm = tf.nn.l2_normalize(desc_db, axis=1)
        
        # Find nearest neighbor matches based on descriptors
        descriptor_distances = self._compute_descriptor_distances(desc_query_norm, desc_db_norm)
        nn_distances = tf.reduce_min(descriptor_distances, axis=1)  # [N_query]
        nn_indices = tf.argmin(descriptor_distances, axis=1)       # [N_query]
        
        # Filter matches by descriptor distance threshold
        valid_descriptor_matches = nn_distances <= self.descriptor_threshold
        
        # Get proposed matches (keypoint correspondences)
        valid_nn_indices = tf.boolean_mask(nn_indices, valid_descriptor_matches)
        valid_query_indices = tf.boolean_mask(
            tf.range(tf.shape(kpts_query)[0]), valid_descriptor_matches
        )
        
        n_proposed_matches = tf.shape(valid_query_indices)[0]
        
        if n_proposed_matches == 0:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Get coordinates of proposed matches
        matched_query_kpts = tf.gather(kpts_query, valid_query_indices)
        matched_db_kpts = tf.gather(kpts_db, valid_nn_indices)
        
        # Transform query keypoints to database coordinate system
        transformed_query_kpts = self._transform_points(matched_query_kpts, homography)
        
        # Compute geometric distances
        geometric_distances = tf.norm(
            transformed_query_kpts - matched_db_kpts, axis=1
        )
        
        # Count geometrically correct matches
        geometrically_correct = geometric_distances <= self.geometric_threshold
        n_correct_matches = tf.reduce_sum(tf.cast(geometrically_correct, tf.float32))
        
        # Compute matching score
        matching_score = n_correct_matches / tf.cast(n_proposed_matches, tf.float32)
        
        return matching_score
    
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
        points_transformed_homo = tf.matmul(points_homo, homography, transpose_b=True)
        
        # Convert back to 2D coordinates
        points_transformed = points_transformed_homo[:, :2] / tf.expand_dims(
            points_transformed_homo[:, 2], axis=1
        )
        
        return points_transformed
    
    def _compute_descriptor_distances(self, desc1, desc2):
        """
        Compute pairwise Euclidean distances between descriptors.
        
        Args:
            desc1: First set of descriptors [N1, D]
            desc2: Second set of descriptors [N2, D]
            
        Returns:
            Distance matrix [N1, N2]
        """
        desc1_expanded = tf.expand_dims(desc1, axis=1)  # [N1, 1, D]
        desc2_expanded = tf.expand_dims(desc2, axis=0)  # [1, N2, D]
        
        squared_distances = tf.reduce_sum(
            tf.square(desc1_expanded - desc2_expanded), axis=2
        )
        
        distances = tf.sqrt(squared_distances + 1e-12)
        
        return distances
    
    def result(self):
        """Return the current metric result."""
        return tf.math.divide_no_nan(self.total_score, self.count)
    
    def reset_state(self):
        """Reset all metric state variables."""
        self.total_score.assign(0.0)
        self.count.assign(0.0)
