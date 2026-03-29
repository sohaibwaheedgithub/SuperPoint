import keras 
import tensorflow as tf
from constants import SP_INPUT_SHAPE


class NearestNeighborMAP(keras.metrics.Metric):
    """
    Nearest Neighbor mean Average Precision metric for SuperPoint descriptors.
    
    This metric evaluates descriptor discriminativeness by computing the Area Under Curve
    of Precision-Recall curves using Nearest Neighbor matching strategy.
    """
    
    def __init__(self, 
                 geometric_threshold=3.0,
                 descriptor_thresholds=None,
                 cell_size=8,
                 name='nn_map', 
                 **kwargs):
        """
        Args:
            geometric_threshold (float): Distance threshold for geometric matching (pixels)
            descriptor_thresholds (list): Descriptor distance thresholds for P-R curve
            cell_size (int): Cell size for converting feature map coords to image coords
            name (str): Name of the metric
        """
        super(NearestNeighborMAP, self).__init__(name=name, **kwargs)
        
        self.geometric_threshold = geometric_threshold
        self.cell_size = cell_size
        
        # Default descriptor thresholds if not provided
        if descriptor_thresholds is None:
            self.descriptor_thresholds = tf.range(0.0, 3.5, 0.5)
        else:
            self.descriptor_thresholds = descriptor_thresholds
            
        # Initialize state variables
        self.total_map = self.add_weight(
            name='total_map', initializer='zeros', dtype=tf.float32
        )
        self.count = self.add_weight(
            name='count', initializer='zeros', dtype=tf.float32
        )
    
    @tf.function(
        input_signature=(
            tf.TensorSpec([SP_INPUT_SHAPE[0] // 8, SP_INPUT_SHAPE[1] // 8, 256], tf.float32),
            tf.TensorSpec([SP_INPUT_SHAPE[0] // 8, SP_INPUT_SHAPE[1] // 8, 256], tf.float32),
            tf.TensorSpec([3, 3], tf.float32)
        ),
        jit_compile=False
    )
    def update_state(self, descriptors1, descriptors2, homography):
        """
        Update the metric state.
        
        Args:
            - 'descriptors1': Dense descriptors from first image [H, W, 256]
            - 'descriptors2': Dense descriptors from second image [H, W, 256]
            - 'homography': Homography matrix [3, 3]
        """
        
        # Compute mAP for this pair
        map_score = self._compute_map(descriptors1, descriptors2, homography)
                    
        self.total_map.assign_add(map_score)
        self.count.assign_add(1.)
    
    def _compute_map(self, desc1, desc2, homography):
        """
        Compute mean Average Precision for a pair of descriptor maps.
        
        Args:
            desc1: Descriptors from first image [H, W, 256]
            desc2: Descriptors from second image [H, W, 256]
            homography: Homography matrix [3, 3]
            
        Returns:
            Mean Average Precision score
        """
        
        # Flatten descriptors and get coordinates
        desc1_flat, coords1 = self._flatten_descriptors(desc1)  # [N1, 256], [N1, 2]
        desc2_flat, coords2 = self._flatten_descriptors(desc2)  # [N2, 256], [N2, 2]
        
        # L2 normalize descriptors
        desc1_norm = tf.nn.l2_normalize(desc1_flat, axis=1)
        desc2_norm = tf.nn.l2_normalize(desc2_flat, axis=1)
        
        # Transform coordinates from image1 to image2 space
        coords1_transformed = self._transform_coordinates(coords1, homography)
        
        # Compute ground truth matches based on geometric distance
        gt_matches_12 = self._compute_ground_truth_matches(coords1_transformed, coords2)
        gt_matches_21 = self._compute_ground_truth_matches(coords2, coords1_transformed)
        
        # Compute mAP in both directions
        map_12 = self._compute_directional_map(desc1_norm, desc2_norm, gt_matches_12)
        map_21 = self._compute_directional_map(desc2_norm, desc1_norm, gt_matches_21)
        
        # Average both directions
        return (map_12 + map_21) / 2.0
    
    def _flatten_descriptors(self, descriptors):
        """
        Flatten descriptor map and get corresponding coordinates.
        
        Args:
            descriptors: Dense descriptors [H, W, 256]
            
        Returns:
            flattened_descriptors: [H*W, 256]
            coordinates: [H*W, 2] in image pixel coordinates
        """
        H, W = SP_INPUT_SHAPE[0] // 8, SP_INPUT_SHAPE[1] // 8
        
        # Flatten descriptors
        desc_flat = tf.reshape(descriptors, [H * W, 256])
        
        # Create coordinate grid
        y_coords, x_coords = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        
        # Convert to image coordinates (multiply by cell_size and add offset)
        x_coords = tf.cast(x_coords, tf.float32) * self.cell_size + self.cell_size // 2
        y_coords = tf.cast(y_coords, tf.float32) * self.cell_size + self.cell_size // 2
        
        # Stack coordinates [H, W, 2] -> [H*W, 2]
        coords = tf.stack([x_coords, y_coords], axis=-1)
        coords_flat = tf.reshape(coords, [H * W, 2])
        
        return desc_flat, coords_flat
    
    def _transform_coordinates(self, coords, homography):
        """
        Transform coordinates using homography matrix.
        
        Args:
            coords: Coordinates [N, 2]
            homography: Homography matrix [3, 3]
            
        Returns:
            Transformed coordinates [N, 2]
        """
        # Convert to homogeneous coordinates
        ones = tf.ones([tf.shape(coords)[0], 1], dtype=coords.dtype)
        coords_homo = tf.concat([coords, ones], axis=1)  # [N, 3]
        
        # Apply transformation
        coords_transformed_homo = tf.matmul(coords_homo, homography, transpose_b=True)
        
        # Convert back to 2D coordinates
        coords_transformed = coords_transformed_homo[:, :2] / tf.expand_dims(
            coords_transformed_homo[:, 2], axis=1
        )
        
        return coords_transformed
    
    def _compute_ground_truth_matches(self, coords1, coords2):
        """
        Compute ground truth matches based on geometric distance.
        
        Args:
            coords1: Coordinates from first set [N1, 2]
            coords2: Coordinates from second set [N2, 2]
            
        Returns:
            Ground truth match indices [N1] (-1 if no match)
        """
        # Compute pairwise distances
        coords1_expanded = tf.expand_dims(coords1, axis=1)  # [N1, 1, 2]
        coords2_expanded = tf.expand_dims(coords2, axis=0)  # [1, N2, 2]

        
        distances = tf.norm(coords1_expanded - coords2_expanded, axis=2)  # [N1, N2]
        
        # Find nearest neighbor for each point in coords1
        nearest_distances = tf.reduce_min(distances, axis=1)  # [N1]
        nearest_indices = tf.argmin(distances, axis=1)  # [N1]

        
        # Only keep matches within geometric threshold
        valid_matches = nearest_distances <= self.geometric_threshold
        gt_matches = tf.where(valid_matches, nearest_indices, -1)
        
        return gt_matches
    
    def _compute_directional_map(self, desc_query, desc_database, gt_matches):
        """
        Compute mAP in one direction (query -> database).
        
        Args:
            desc_query: Query descriptors [N_query, 256]
            desc_database: Database descriptors [N_db, 256]
            gt_matches: Ground truth matches [N_query] (-1 if no match)
            
        Returns:
            Mean Average Precision score
        """
        # Compute descriptor distances
        desc_distances = self._compute_descriptor_distances(desc_query, desc_database)
        
        # Find nearest neighbor matches
        nn_distances = tf.reduce_min(desc_distances, axis=1)  # [N_query]
        nn_indices = tf.argmin(desc_distances, axis=1)  # [N_query]
        
        # Compute precision-recall for different thresholds
        precisions = tf.TensorArray(tf.float32, size=tf.shape(self.descriptor_thresholds)[0], element_shape=())
        recalls = tf.TensorArray(tf.float32, size=tf.shape(self.descriptor_thresholds)[0], element_shape=())
        
        
        def loop_body(i, precisions_ta, recalls_ta):
            threshold = self.descriptor_thresholds[i]
            # Positive predictions: NN distance <= threshold
            positive_mask = nn_distances <= threshold
            
            # True positives: positive predictions that match ground truth
            true_positive_mask = tf.logical_and(
                positive_mask,
                tf.equal(nn_indices, tf.maximum(gt_matches, 0))
            )
            
            # Only consider queries that have ground truth matches
            valid_gt_mask = gt_matches >= 0
            
            # Compute precision and recall
            n_true_positives = tf.reduce_sum(tf.cast(true_positive_mask, tf.float32))
            n_positives = tf.reduce_sum(tf.cast(positive_mask, tf.float32))
            n_ground_truth = tf.reduce_sum(tf.cast(valid_gt_mask, tf.float32))
            
            precision = tf.math.divide_no_nan(n_true_positives, n_positives)
            recall = tf.math.divide_no_nan(n_true_positives, n_ground_truth)
            
            precisions_ta = precisions_ta.write(i, precision)
            recalls_ta = recalls_ta.write(i, recall)

            return i+1, precisions_ta, recalls_ta
        
        _, precisions, recalls = tf.while_loop(
            cond=lambda i, *_: i < tf.shape(self.descriptor_thresholds)[0],
            body=loop_body,
            loop_vars=[0, precisions, recalls]
        )
        
        # Compute Average Precision using trapezoidal rule
        precisions = precisions.stack()
        recalls = recalls.stack()
        
        
        # Sort by recall
        sorted_indices = tf.argsort(recalls)
        sorted_recalls = tf.gather(recalls, sorted_indices)
        sorted_precisions = tf.gather(precisions, sorted_indices)
        
        # Compute AUC using trapezoidal rule
        recall_diffs = sorted_recalls[1:] - sorted_recalls[:-1]
        precision_avgs = (sorted_precisions[1:] + sorted_precisions[:-1]) / 2.0
        
        ap = tf.reduce_sum(recall_diffs * precision_avgs)
        
        return ap
    
    def _compute_descriptor_distances(self, desc1, desc2):
        """
        Compute pairwise Euclidean distances between descriptors.
        
        Args:
            desc1: First set of descriptors [N1, D]
            desc2: Second set of descriptors [N2, D]
            
        Returns:
            Distance matrix [N1, N2]
        """
        # Expand dimensions for broadcasting
        desc1_expanded = tf.expand_dims(desc1, axis=1)  # [N1, 1, D]
        desc2_expanded = tf.expand_dims(desc2, axis=0)  # [1, N2, D]
        
        # Compute squared distances
        squared_distances = tf.reduce_sum(
            tf.square(desc1_expanded - desc2_expanded), axis=2
        )  # [N1, N2]
        
        # Take square root to get Euclidean distances
        distances = tf.sqrt(squared_distances + 1e-12)  # Add small epsilon for numerical stability
        
        return distances
    
    def result(self):
        """Return the current metric result."""
        return tf.math.divide_no_nan(self.total_map, self.count)
    
    def reset_state(self):
        """Reset all metric state variables."""
        self.total_map.assign(0.0)
        self.count.assign(0.0)
