import keras
import tensorflow as tf
from superpoint import SP_INPUT_SHAPE

class DescriptorLoss(keras.losses.Loss):
    """
    Keras implementation of the SuperPoint descriptor loss function.
    
    This loss compares descriptors between original and transformed images
    using the homography mapping to determine positive and negative correspondences.
    """
    
    def __init__(self, 
        positive_margin=1.0, 
        negative_margin=0.2, 
        delta=250,
        name='superpoint_descriptor_loss'
    ):
        """
        Initialize SuperPoint descriptor loss.
        
        Args:
            original_image_height: Height of the original input image
            original_image_width: Width of the original input image
            positive_margin: Margin for positive pairs (mp)
            negative_margin: Margin for negative pairs (mn)
            delta: Weight factor to balance positive and negative pairs
            name: Name of the loss function
        """
        super().__init__(name=name)
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.delta = delta
        
        # Pre-compute static values that don't change between calls
        # Assuming feature map dimensions are 30x40 (8x downsampling)
        self.Hc = SP_INPUT_SHAPE[0] // 8
        self.Wc = SP_INPUT_SHAPE[1] // 8
        
        # Create cell center coordinates grid for the first descriptor map
        h_indices = tf.range(self.Hc, dtype=tf.float32)
        w_indices = tf.range(self.Wc, dtype=tf.float32)
        
        # Compute pixel coordinates for each cell center
        h_coords = (h_indices + 0.5) * 8
        w_coords = (w_indices + 0.5) * 8
        
        # Create meshgrid of coordinates
        grid_w, grid_h = tf.meshgrid(w_coords, h_coords)
        
        # Reshape to [H*W, 2] and add ones to get homogeneous coordinates [x, y, 1]
        grid_hw = tf.stack([grid_w, grid_h], axis=-1)
        grid_hw = tf.reshape(grid_hw, [-1, 2])
        ones = tf.ones([tf.shape(grid_hw)[0], 1], dtype=tf.float32)
        self.grid_hw_homogeneous = tf.concat([grid_hw, ones], axis=-1)  # [H*W, 3]
        
        # Create all possible cell coordinates for the second image
        h2_grid, w2_grid = tf.meshgrid(h_indices, w_indices, indexing='ij')
        cell_coords_2 = tf.stack([w2_grid, h2_grid], axis=-1)  # [H, W, 2]
        cell_coords_2 = tf.reshape(cell_coords_2, [self.Hc*self.Wc, 2])  # [H*W, 2]
        
        # Convert to pixel coordinates
        self.pixel_coords_2 = tf.stack([
            (cell_coords_2[:, 0] + 0.5) * 8,  # x-coordinates
            (cell_coords_2[:, 1] + 0.5) * 8   # y-coordinates
        ], axis=1)  # [H*W, 2]
        
    def call(self, y_true, y_pred):
        """
        Calculate the SuperPoint descriptor loss.
        
        Args:
            y_true: A tuple containing the homography matrix with shape [batch_size, 3, 3]
            y_pred: A tuple containing (desc1, desc2) where:
                    - desc1: Descriptor map from first image with shape [batch_size, H, W, D]
                    - desc2: Descriptor map from second image with shape [batch_size, H, W, D]
        
        Returns:
            The descriptor loss value
        """
        # Unpack inputs
        homography = y_true
        desc1, desc2 = y_pred
        
        # Function to process a single batch element
        def process_batch_element(inputs):
            desc1_b, desc2_b, H_b = inputs
            
            # Apply homography to center coordinates
            # Transform points: [H*W, 3] x [3, 3] = [H*W, 3]
            transformed_points = tf.matmul(self.grid_hw_homogeneous, tf.transpose(H_b))
            
            # Convert from homogeneous to Euclidean coordinates
            transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]  # [H*W, 2]
            
            # Flatten descriptors
            desc1_flat = tf.reshape(desc1_b, [self.Hc*self.Wc, 256])  # [H*W, D]
            desc2_flat = tf.reshape(desc2_b, [self.Hc*self.Wc, 256])  # [H*W, D]
            
            # Compute all pairwise dot products
            dot_products = tf.matmul(desc1_flat, desc2_flat, transpose_b=True)  # [H*W, H*W]
            
            # Create correspondence matrix based on Euclidean distance threshold
            # According to the paper: s_hw,h'w' = 1 if ||H·p_hw - p_h'w'|| ≤ 8, otherwise 0
            
            # Compute pairwise distances between transformed points and all cell centers in second image
            # Reshape transformed_points to [H*W, 1, 2] for broadcasting
            transformed_points_expanded = tf.expand_dims(transformed_points[:, :2], 1)  # [H*W, 1, 2]
            # Reshape pixel_coords_2 to [1, H*W, 2] for broadcasting
            pixel_coords_2_expanded = tf.expand_dims(self.pixel_coords_2, 0)  # [1, H*W, 2]
            
            # Compute Euclidean distances: [H*W, H*W]
            distances = tf.sqrt(tf.reduce_sum(
                tf.square(transformed_points_expanded - pixel_coords_2_expanded), 
                axis=2
            ))
            
            # Create correspondence matrix where distance ≤ 8 pixels
            threshold = 8.0  # As specified in the paper
            correspondence_matrix = tf.cast(distances <= threshold, tf.float32)
            
            # Apply the loss
            # Positive pairs: s=1, want dot product to be high (>= mp)
            positive_loss = self.delta * correspondence_matrix * tf.maximum(0.0, self.positive_margin - dot_products)
            
            # Negative pairs: s=0, want dot product to be low (<= mn)
            negative_loss = (1.0 - correspondence_matrix) * tf.maximum(0.0, dot_products - self.negative_margin)
            
            # Combine and normalize
            total = tf.reduce_sum(positive_loss + negative_loss) / tf.cast((self.Hc*self.Wc)*(self.Hc*self.Wc), tf.float32)
            return total
        
        # Map the function across the batch
        batch_losses = tf.map_fn(
            process_batch_element,
            (desc1, desc2, homography),
            fn_output_signature=tf.float32
        )
        
        # Return the mean across the batch
        return tf.reduce_mean(batch_losses)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'positive_margin': self.positive_margin,
            'negative_margin': self.negative_margin,
            'delta': self.delta
        })
        return config

