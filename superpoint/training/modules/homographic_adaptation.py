import tensorflow as tf
from constants import MP_INPUT_SHAPE, MP_BATCH_SIZE


# Homographic adaptation for generating pseudo ground truth
class HomographicAdapter:
    def __init__(self, n_homographies=100):
        """
        Initialize the homographic adapter
        
        Args:
            n_homographies: Number of homographies to use for each image
        """
        self.n_homographies = n_homographies
        self.interest_point_model = None  # Will be set by the trainer
        
    def set_interest_point_model(self, model):
        """
        Set the interest point detection model to use for pseudo label generation
        
        Args:
            model: SuperPoint model or any model with interest point detection capability
        """
        self.interest_point_model = model
    
    @tf.function(jit_compile=True)    
    def meshgrid(self):
        x_t = tf.linspace(0.0, tf.cast(MP_INPUT_SHAPE[1] - 1, tf.float32), MP_INPUT_SHAPE[1])
        y_t = tf.linspace(0.0, tf.cast(MP_INPUT_SHAPE[0] - 1, tf.float32), MP_INPUT_SHAPE[0])
        x_t, y_t = tf.meshgrid(x_t, y_t)
        ones = tf.ones_like(x_t)
        grid = tf.stack([x_t, y_t, ones], axis=0)  # [3, H, W]
        return grid

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[1]+MP_INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=[3, 3], dtype=tf.float32)
        ),
        jit_compile=True
    )    
    def apply_homography(self, image, H):
        H = tf.cast(H, tf.float32)
        batch_size = tf.shape(image)[0]
        height, width = MP_INPUT_SHAPE[0], MP_INPUT_SHAPE[1]

        grid = self.meshgrid()  # [3, H, W]
        grid_flat = tf.reshape(grid, [3, -1])  # [3, H*W]

        H_inv = tf.linalg.inv(H)  # [B, 3, 3]
        H_inv = tf.reshape(H_inv, [-1, 3, 3])

        # Tile the grid for batch processing
        grid_flat = tf.expand_dims(grid_flat, axis=0)
        grid_flat = tf.tile(grid_flat, [batch_size, 1, 1])  # [B, 3, H*W]

        warped_coords = tf.matmul(H_inv, grid_flat)  # [B, 3, H*W]
        x_warped = warped_coords[:, 0, :] / (warped_coords[:, 2, :] + 1e-8)
        y_warped = warped_coords[:, 1, :] / (warped_coords[:, 2, :] + 1e-8)

        x0 = tf.floor(x_warped)
        x1 = x0 + 1
        y0 = tf.floor(y_warped)
        y1 = y0 + 1

        x0_safe = tf.clip_by_value(x0, 0.0, tf.cast(width - 1, tf.float32))
        x1_safe = tf.clip_by_value(x1, 0.0, tf.cast(width - 1, tf.float32))
        y0_safe = tf.clip_by_value(y0, 0.0, tf.cast(height - 1, tf.float32))
        y1_safe = tf.clip_by_value(y1, 0.0, tf.cast(height - 1, tf.float32))

        def get_pixel_value(img, x, y):
            b = tf.range(batch_size)
            b = tf.reshape(b, [batch_size, 1])
            b = tf.tile(b, [1, tf.shape(x)[1]])  # [B, N]

            indices = tf.stack([b, tf.cast(y, tf.int32), tf.cast(x, tf.int32)], axis=-1)  # [B, N, 3]
            return tf.gather_nd(img, indices)

        Ia = get_pixel_value(image, x0_safe, y0_safe)
        Ib = get_pixel_value(image, x0_safe, y1_safe)
        Ic = get_pixel_value(image, x1_safe, y0_safe)
        Id = get_pixel_value(image, x1_safe, y1_safe)

        wa = (x1 - x_warped) * (y1 - y_warped)
        wb = (x1 - x_warped) * (y_warped - y0)
        wc = (x_warped - x0) * (y1 - y_warped)
        wd = (x_warped - x0) * (y_warped - y0)

        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        warped_image = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [B, H*W, C]
        warped_image = tf.reshape(warped_image, [batch_size, height, width, -1])
        return warped_image
    
    @tf.function(input_signature=(tf.TensorSpec(shape=MP_INPUT_SHAPE, dtype=tf.float32),), jit_compile=True)
    def random_homographic_transform(self, image, params=None):
        """
        Apply a random homographic transformation to an image
        
        Args:
            image: Input image tensor
            params: Optional dictionary of transformation parameters
                   If None, default medium intensity parameters are used
                   
        Returns:
            transformed_image: The warped image
            H: The homography matrix
            H_inverse: The inverse homography matrix
        """
        if params is None:
            params = {
                'crop_ratio_range': (0.95, 0.98),
                'translation_range': (-2.0, 2.0),
                'scale_range': (0.95, 1.05),
                'rotation_range': (-5.0, 5.0),
                'perspective_range': (-0.00005, 0.00005)
            }
        
        # Extract parameters
        crop_ratio_range = params['crop_ratio_range']
        translation_range = params['translation_range']
        scale_range = params['scale_range']
        rotation_range = params['rotation_range']
        perspective_range = params['perspective_range']
        
        # Random crop ratio
        crop_ratio = tf.random.uniform(
            [], minval=crop_ratio_range[0], maxval=crop_ratio_range[1]
        )
        
        # Random translation values
        tx = tf.random.uniform(
            [], minval=translation_range[0], maxval=translation_range[1]
        )
        ty = tf.random.uniform(
            [], minval=translation_range[0], maxval=translation_range[1]
        )
        
        # Random scaling factor
        scale = tf.random.uniform(
            [], minval=scale_range[0], maxval=scale_range[1]
        )
        
        # Random rotation angle in degrees
        angle_deg = tf.random.uniform(
            [], minval=rotation_range[0], maxval=rotation_range[1]
        )
        
        # Convert to radians
        angle_rad = angle_deg * tf.experimental.numpy.pi / 180.0
        
        # Create rotation matrix
        cos_val = tf.cos(angle_rad)
        sin_val = tf.sin(angle_rad)
        
        # Random perspective strength
        strength_1 = tf.random.uniform(
            [], minval=perspective_range[0], maxval=perspective_range[1]
        )
        strength_2 = tf.random.uniform(
            [], minval=perspective_range[0], maxval=perspective_range[1]
        )
        
        # Create perspective transform matrix
        random_homography_matrix = [
            [scale*cos_val, -scale*sin_val, tx],
            [scale*sin_val, scale*cos_val, ty],
            [strength_1, strength_2, crop_ratio]
        ]
        
        # Convert to tensor
        H = tf.convert_to_tensor(random_homography_matrix, dtype=tf.float32)
        H_inverse = tf.linalg.inv(H)
        
        # Apply transformation to image
        '''transformed_img = tf.numpy_function(
            lambda img, matrix: cv2.warpPerspective(
                img,
                matrix,
                dsize=MP_INPUT_SHAPE[-2::-1],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ),
            [image, H],
            tf.float32
        )'''
        
        transformed_img = self.apply_homography(image[tf.newaxis, ...], H)
            
        return transformed_img, H, H_inverse
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),), jit_compile=True)    
    def generateBins(self, points):
        points = tf.round(points)
        # To prepare all possible set of coordinates of points in the image
        x = range(0, MP_INPUT_SHAPE[0])
        y = range(0, MP_INPUT_SHAPE[1])
        X, Y = tf.meshgrid(x, y, indexing="ij")
        # Shaping it up in this form so that points can be compared using tf.equal
        X, Y = X[..., tf.newaxis], Y[..., tf.newaxis]
        gridsRegion = tf.reshape(tf.cast(tf.concat([X, Y], axis=-1), tf.float32), [-1, 1, 2])
        # Comparing each coordinate position with all ground truth points to get a tensor of shape [total_cooridnates, n_gt_pts, 2]
        # For a point to lie on a pixel, both of it's coordinates should match with pixel's both coordinates i.e [True, True]
        # Then reducing [bool, bool] -> [bool] to get only those pixels where both coordinates match
        binsBooleanMask = tf.reduce_all(tf.equal(gridsRegion, points[tf.newaxis, ...]), axis=-1)
        # Reshaping [total_cooridnates, n_gt_pts] -> [120, 160, n_gt_pts]
        binsBooleanMask = tf.reshape(binsBooleanMask, [MP_INPUT_SHAPE[0], MP_INPUT_SHAPE[1], -1])
        # converting True -> 1 and False -> 0, since amoung all points there exists only one point that lies on a certain
        # pixel, then if we sum all points together we will get 1 for pixels where points lie and 0 for pixels where points
        # doesn't lie
        # Also adding and batch dimension and depth dimension as tf.nn.space_to_depth expects so
        binsBinaryMask = tf.reduce_sum(tf.cast(binsBooleanMask, tf.float32), axis=-1)[tf.newaxis, ..., tf.newaxis]
        # Now extracting patches of size 30 x 40 x 64 from the image by sliding 8 x 8 window (going from space to depth)
        bins = tf.nn.space_to_depth(binsBinaryMask, block_size=8)[0]
        bins = tf.concat([bins, tf.ones_like(bins)[..., -1:]*0.5], axis=-1)
        bins = tf.argmax(bins, axis=-1, output_type=tf.int32)
        return bins
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[MP_BATCH_SIZE]+MP_INPUT_SHAPE, dtype=tf.float32),), jit_compile=True)
    def generate_data(self, batch_images):
        """
        Generate pseudo ground truth for a batch of images using Homographic Adaptation
        
        Args:
            batch_images: Batch of images [batch_size, height, width, channels]
            
        Returns:
            pseudo_labels: Tensor of pseudo ground truth heatmaps
        """
        
        # Preallocate arrays for results
        pseudo_labels = tf.TensorArray(tf.float32, size=MP_BATCH_SIZE, element_shape=MP_INPUT_SHAPE)
        pseudo_bins = tf.TensorArray(tf.int32, size=MP_BATCH_SIZE, element_shape=[MP_INPUT_SHAPE[0]//8, MP_INPUT_SHAPE[1]//8])
        transformed_images = tf.TensorArray(tf.float32, size=MP_BATCH_SIZE, element_shape=MP_INPUT_SHAPE)
        transformed_labels = tf.TensorArray(tf.float32, size=MP_BATCH_SIZE, element_shape=MP_INPUT_SHAPE[:-1])
        transformed_bins = tf.TensorArray(tf.int32, size=MP_BATCH_SIZE, element_shape=[MP_INPUT_SHAPE[0]//8, MP_INPUT_SHAPE[1]//8])
        homography_matrices = tf.TensorArray(tf.float32, size=MP_BATCH_SIZE, element_shape=[3, 3])
        
        # Process each image in the batch
        for i in tf.range(MP_BATCH_SIZE):
            img = batch_images[i]
            
            # Apply homographic adaptation to get robust points using current model
            final_output = self.homographic_adaptation(img)
            labels = final_output[..., tf.newaxis]
            bins = self.generateBins(tf.cast(tf.sparse.from_dense(labels).indices[:, :-1], tf.float32))
            
            # Apply random homographic transformations to the training images and their pseudo-labels
            transformed_img, H, _ = self.random_homographic_transform(img)
            
            # Transform pseudo-label to match the transformed image
            '''transformed_label = tf.numpy_function(
                lambda lbl, matrix: cv2.warpPerspective(
                    lbl,
                    matrix,
                    dsize=MP_INPUT_SHAPE[-2::-1],
                    flags=cv2.INTER_NEAREST,  # Use nearest for labels to avoid interpolation issues
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                ),
                [labels, H],
                tf.float32
            )'''
            
            transformed_label = self.apply_homography(labels[tf.newaxis, ...], H)[0, ..., 0]
            transformed_bin = self.generateBins(
                tf.cast(tf.sparse.from_dense(transformed_label).indices, tf.float32)
            )

            # Store the result
            pseudo_labels = pseudo_labels.write(i, labels)
            pseudo_bins = pseudo_bins.write(i, bins)
            transformed_images = transformed_images.write(i, transformed_img[0])
            transformed_labels = transformed_labels.write(i, transformed_label)
            transformed_bins = transformed_bins.write(i, transformed_bin)
            homography_matrices = homography_matrices.write(i, H)
            
        # Stack all results into a batch
        return {
            "pseudo_labels": pseudo_labels.stack(),
            "pseudo_bins": pseudo_bins.stack(),
            "transformed_images": transformed_images.stack(),
            "transformed_labels": transformed_labels.stack(),
            "transformed_bins": transformed_bins.stack(),
            "homography_matrices": homography_matrices.stack()
        }
    
    @tf.function(input_signature=(tf.TensorSpec(shape=MP_INPUT_SHAPE, dtype=tf.float32),), jit_compile=True)
    def homographic_adaptation(self, img, threshold=0.015):
        """
        Apply homographic adaptation to get robust interest points using the current model
        
        Args:
            img: Input image tensor [height, width, channels]
            threshold: Threshold for considering a point as an interest point
            
        Returns:
            final_output: Averaged and thresholded interest point heatmap
        """
        # Storage for outputs
        unwarped_outputs = tf.TensorArray(tf.float32, size=self.n_homographies, element_shape=MP_INPUT_SHAPE[:-1])
        
        for i in tf.range(self.n_homographies):
            # Apply random homographic transformation
            transformed_img, H, H_inverse = self.random_homographic_transform(img)
            
            # Run inference on transformed image using current SuperPoint model
            # Use only the interest point detection part (ipdPostProcessedOutput)
            # output = tf.squeeze(
            #     self.interest_point_model(transformed_img, training=False)["detector_post_processor"], 
            #     axis=0
            # )

            output = self.interest_point_model(transformed_img, training=False)["detector_post_processor"]
            
            # Unwarp the detection back to original image space
            '''unwarped_output = tf.numpy_function(
                lambda img, matrix: cv2.warpPerspective(
                    img,
                    matrix,
                    dsize=MP_INPUT_SHAPE[-2::-1],
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                ),
                [output, H_inverse],
                tf.float32
            )'''
        
            #unwarped_output = self.apply_homography(output[tf.newaxis, ...], H_inverse)[0, ..., 0]
            unwarped_output = self.apply_homography(output, H_inverse)[0, ..., 0]
            
            # Store unwarped output
            unwarped_outputs = unwarped_outputs.write(i, unwarped_output)
            
        # Stack and average all outputs
        final_outputs = unwarped_outputs.stack()
        final_output = tf.reduce_mean(final_outputs, axis=0)
        
        # Apply threshold to get binary interest point map
        return tf.cast(tf.greater_equal(final_output, threshold), tf.float32)