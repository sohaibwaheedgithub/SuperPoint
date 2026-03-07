import keras
import tensorflow as tf
from constants import MP_INPUT_SHAPE
from superpoint.training.modules.homographic_adaptation import HomographicAdapter


class SuperPointTrainer:
    def __init__(self, dataset_loader, pretrained_magicpoint_path, mean, variance, validation_dataset=None):
        """
        Initialize the SuperPoint trainer
        
        Args:
            dataset_loader: Dataset loader object for training data
            pretrained_magicpoint_path: Path to pretrained MagicPoint weights
            mean: Mean value for normalization
            variance: Variance value for normalization
            validation_dataset: Optional validation dataset for homography evaluation
        """
        self.dataset_loader = dataset_loader
        self.validation_dataset = validation_dataset
        
        # Load pretrained MagicPoint for initial weight transfer
        self.magic_point = keras.models.load_model(pretrained_magicpoint_path)
        
        # Create SuperPoint model
        self.super_point = build_superpoint(MP_INPUT_SHAPE, mean, variance)
        
        # Initialize homographic adapter for pseudo-label generation
        # Initially use SuperPoint itself for pseudo label generation
        self.homographic_adapter = HomographicAdapter()
        self.homographic_adapter.set_interest_point_model(self.super_point)
        
        # Initialize metrics
        self.repeatability_metric = RepeatabilityMetric(distance_threshold=eta)
        self.nearest_neighbor_map_metric = NearestNeighborMAPMetric(geometric_threshold=eta)
        self.matching_score_metric = MatchingScoreMetric(geometric_threshold=eta)
        self.homography_estimation_metric = HomographyEstimationMetric(correctness_threshold=eta)

        # For writing training summary
        self.train_summary_writer = tf.summary.create_file_writer("superpoint_logs/train")
        self.valid_summary_writer = tf.summary.create_file_writer("superpoint_logs/valid")
        
        # Copy weights from MagicPoint to SuperPoint (shared encoder and interest point decoder)
        self._transfer_weights()
        
        # Compile the model with both detector and descriptor losses
        self._compile_model()
        
    def _transfer_weights(self):
        """Transfer weights from MagicPoint to SuperPoint for the shared components"""
        # Extract weights from MagicPoint
        for layer in self.magic_point.layers:
            if isinstance(layer, SharedEncoder) or isinstance(layer, InterestPointDecoder) or isinstance(layer, IPDPostProcessor):
                # Find corresponding layer in SuperPoint
                for sp_layer in self.super_point.layers:
                    if type(sp_layer) == type(layer):
                        sp_layer.set_weights(layer.get_weights())
                        print(f"Transferred weights for layer: {type(layer).__name__}")
    
    def _compile_model(self):
        """Compile the SuperPoint model with appropriate losses and metrics"""
        
        # Compile with detector loss (binary crossentropy for interest point detection)
        self.super_point.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss = {
                "interestPointDecoderOutput": keras.losses.SparseCategoricalCrossentropy(),
                "descriptorOutput": DescriptorLoss(positive_margin=1.0, negative_margin=0.2, delta=250.0)
            },
            jit_compile=True
        )
    
    def _extract_keypoints_and_descriptors(self, outputs):
        """
        Extract keypoints and descriptors from model outputs
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            keypoints: [N, 2] tensor of keypoint coordinates
            descriptors: [N, D] tensor of descriptors
        """
        # Get interest point predictions
        interest_points = outputs["ipdPostProcessedOutput"][0]  # [H, W]
        descriptors_map = outputs["descriptorOutput"][0]
        dense_descriptors_map = tf.nn.l2_normalize(tf.image.resize(outputs["descriptorOutput"], MP_INPUT_SHAPE[:-1]))[0]   # [H, W, D]
        
        # Find keypoint locations (non-zero pixels)
        keypoint_indices = tf.where(interest_points > detection_threshold)[:, :-1]  # threshold
        
        if tf.shape(keypoint_indices)[0] == 0:
            # No keypoints found, return empty tensors
            return tf.zeros([0, 2], dtype=tf.float32), tf.zeros([0, 256], dtype=tf.float32), tf.zeros([0, 256], dtype=tf.float32)
        
        # Extract keypoint coordinates
        keypoints = tf.cast(keypoint_indices, tf.float32)
        
        # Extract corresponding descriptors
        descriptors = tf.gather_nd(dense_descriptors_map, keypoint_indices)
        
        return keypoints, descriptors, descriptors_map
    
    def _evaluate_metrics(self, num_samples=1):
        
        # Reset all metrics
        self.repeatability_metric.reset_state()            
        self.nearest_neighbor_map_metric.reset_state()
        self.matching_score_metric.reset_state()
        self.homography_estimation_metric.reset_state()
        
        # Sample validation data
        validation_iter = iter(self.validation_dataset.take(num_samples))
        
        for _ in range(num_samples):
            try:
                batch_images = next(validation_iter)
                
                # Generate image pairs with known homography
                for i in range(MP_BATCH_SIZE):  # Process pairs
                    img1 = batch_images[i:i+1]
                    
                    # Apply known homographic transformation
                    img2, H_gt, _ = self.homographic_adapter.random_homographic_transform(img1[0])
                    
                    # Get model predictions for both images
                    outputs1 = self.super_point(img1, training=False)
                    outputs2 = self.super_point(img2, training=False)
                    
                    # Extract keypoints and descriptors
                    kp1, kp1_desc, desc1 = self._extract_keypoints_and_descriptors(outputs1)
                    kp2, kp2_desc, desc2 = self._extract_keypoints_and_descriptors(outputs2)
                    
                    if not tf.logical_or(
                        tf.equal(tf.shape(kp1)[0], 0),
                        tf.equal(tf.shape(kp2)[0], 0)
                    ):
                        # Update Repeatability metric
                        #self.repeatability_metric.update_state(kp1, kp2, H_gt)
                        # Update Nearest Neighbor MAP metric
                        #self.nearest_neighbor_map_metric.update_state(desc1, desc2, H_gt)
                        # Update Matching score metric
                        #self.matching_score_metric.update_state(kp1, kp1_desc, kp2, kp2_desc, H_gt)  
                        # Update homography metric
                        self.homography_estimation_metric.update_state(kp1, kp1_desc, kp2, kp2_desc, H_gt)
                    
            except StopIteration:
                break
                
        return {
            "Rep": self.repeatability_metric.result().numpy(),
            "NN MAP": self.nearest_neighbor_map_metric.result().numpy(),
            "M Score": self.matching_score_metric.result().numpy(),
            "Homo Est": self.homography_estimation_metric.result().numpy()
        }
       
    def train(self, epochs, steps_per_epoch, evaluate_metrics_every=5, update_pseudo_labels_every=1):
        """
        Train the SuperPoint model with self-supervised pseudo label generation
        
        Args:
            epochs: Number of training epochs
            steps_per_epoch: Number of steps per epoch
            evaluate_metrics_every: Evaluate metrics every N epochs
            update_pseudo_labels_every: Update pseudo labels using current model every N epochs
            
        Returns:
            history: Training history
        """
        # Create dataset iterator
        dataset = self.dataset_loader.load_dataset('train')
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Update pseudo label generation model periodically
            # This is the key improvement: use updated SuperPoint model for better pseudo labels
            if epoch > 0 and (epoch % update_pseudo_labels_every == 0):
                print("Updating pseudo label generation model with current SuperPoint weights...")
                # The homographic adapter already uses self.super_point, so it automatically uses updated weights
                # This creates a self-supervised learning loop where better detectors generate better pseudo labels
            
            # Reset metrics
            for metric in self.super_point.metrics:
                metric.reset_state()
                        
            progress_bar = tqdm(dataset.take(steps_per_epoch), total=steps_per_epoch, desc=f"Epoch {epoch+1}", ncols=100)
            # Training loop
            epoch_metrics = None
            for step, batch_images in enumerate(progress_bar):   
                # Generate pseudo labels using current SuperPoint model (self-supervised improvement)
                generated_data = self.homographic_adapter.generate_data(batch_images)
                # Execute training step
                epoch_metrics = self.super_point.train_step({**generated_data, "batch_images": batch_images})
            
            # Store training metrics
            if epoch_metrics:
                # Writing loss summary
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_metrics['loss'].numpy(), step=epoch)
                    tf.summary.scalar('total_loss', epoch_metrics['total_loss'].numpy(), step=epoch)
                    tf.summary.scalar('detect_loss_1', epoch_metrics['detect_loss_1'].numpy(), step=epoch)
                    tf.summary.scalar('detect_loss_2', epoch_metrics['detect_loss_2'].numpy(), step=epoch)
                    tf.summary.scalar('desc_loss', epoch_metrics['desc_loss'].numpy(), step=epoch)


                #progress_bar.set_postfix(loss=epoch_metrics['loss'].numpy())

            # For epoch summary
            metrics_str = ' - '.join([f"{k}: {v.numpy():.4f}" for k, v in epoch_metrics.items()])
            
            # Evaluate all metrics periodically
            if (epoch + 1) % evaluate_metrics_every == 0:
                metrics_acc = self._evaluate_metrics()
                
                metrics_str += f" - Repeatability: {metrics_acc['Rep']:.4f}"
                metrics_str += f" - Nearest Neighbor MAP: {metrics_acc['NN MAP']:.4f}"
                metrics_str += f" - Matching Score: {metrics_acc['M Score']:.4f}"
                metrics_str += f" - Homography Estimation: {metrics_acc['Homo Est']:.4f}"
                
                # Writing metrics to summary
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("Repeatability", metrics_acc["Rep"], step=epoch)
                    tf.summary.scalar("Nearest Neighbor MAP", metrics_acc["NN MAP"], step=epoch)
                    tf.summary.scalar("Matching Score", metrics_acc["M Score"], step=epoch)
                    tf.summary.scalar("Homography Estimation", metrics_acc["Homo Est"], step=epoch)
                
                print(f"Step {step}/{steps_per_epoch} - {metrics_str}")

            
            # Save checkpoint after each epoch
            self.super_point.save(f"saved_models/superpoint_{epoch+1}.keras")
            
            print(f"Epoch {epoch+1}/{epochs} - {metrics_str}")
