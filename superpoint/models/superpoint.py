import keras
import tensorflow as tf
from superpoint.models.components.encoder import SharedEncoder
from superpoint.models.components.decoder import Decoder
from superpoint.models.components.post_processor import DetectorPostProcessor, DescriptorPostProcessor
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision




class SuperPoint(keras.Model):
    def __init__(self, mean, variance, homographic_adapter=None):
        super().__init__()

        self.shared_encoder = SharedEncoder(name="shared_encoder")
        # Interest Point Decoder
        self.detector_decoder = Decoder(65, name="detector_decoder")
        self.detector_post = DetectorPostProcessor(name="detector_post_processor")
        # Descriptor Decoder
        self.descriptor_decoder = Decoder(256, name="descriptor_decoder")
        self.descriptor_post = DescriptorPostProcessor(name="descriptor_post_processor")

        self.homographic_adapter = homographic_adapter
        self.detector_loss_fn = None
        self.descriptor_loss_fn = None
        self.descriptor_loss_weight = 1.0


        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)


    def compile(self, optimizer, loss, descriptor_loss_weight=1.0, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        self.detector_loss_fn = loss["interestPointDecoderOutput"]
        self.descriptor_loss_fn = loss["descriptorOutput"]
        self.descriptor_loss_weight = descriptor_loss_weight



    def call(self, inputs, training=False):
        x = (inputs - self.mean) / tf.sqrt(self.variance)
        encoder_features = self.shared_encoder(x, training=training)
        
        detector_logits = self.detector_decoder(encoder_features, training=training)
        detector_heatmap = self.detector_post(detector_logits)

        descriptor_logits = self.descriptor_decoder(encoder_features, training=training)

        return {
            "encoder_features": encoder_features,
            "bins": detector_logits,
            "heatmap": detector_heatmap,
            "detector_logits": detector_logits,
            "detector_heatmap": detector_heatmap,
            "descriptor_logits": descriptor_logits,
            "descriptor_map": self.descriptor_post(descriptor_logits),
        }



    def train_step(self, data):
        if self.homographic_adapter is None:
            raise ValueError("SuperPoint.train_step requires a homographic_adapter.")
        
        batch_images = data["image"] if isinstance(data, dict) else data

        with tf.GradientTape() as tape:
            homographic_batch = self.homographic_adapter.generate_data(
                batch_images=batch_images,
                interest_point_model=self,
            )
            outputs = self(batch_images, training=True)
            transformed_outputs = self(homographic_batch["transformed_images"], training=True)

            detector_loss_1 = self.detector_loss_fn(
                homographic_batch["pseudo_bins"],
                outputs["detector_logits"],
            )
            detector_loss_2 = self.detector_loss_fn(
                homographic_batch["transformed_bins"],
                transformed_outputs["detector_logits"],
            )
            descriptor_loss = self.descriptor_loss_fn(
                homographic_batch["homography_matrices"],
                (outputs["descriptor_logits"], transformed_outputs["descriptor_logits"]),
            )

            loss = (
                detector_loss_1
                + detector_loss_2
                + (self.descriptor_loss_weight * descriptor_loss)
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


        return_dict = {
            "loss": loss,
            "detector_loss_1": detector_loss_1,
            "detector_loss_2": detector_loss_2,
            "descriptor_loss": descriptor_loss,
        }
        
        return return_dict



    def test_step(self, data):
        if self.homographic_adapter is None:
            raise ValueError("SuperPoint.test_step requires a homographic_adapter.")

        batch_images = data["image"] if isinstance(data, dict) else data
        
        homographic_batch = self.homographic_adapter.generate_data(
            batch_images=batch_images,
            interest_point_model=self,
        )
        outputs = self(batch_images, training=False)
        transformed_outputs = self(homographic_batch["transformed_images"], training=False)

        detector_loss_1 = self.detector_loss_fn(
            homographic_batch["pseudo_bins"],
            outputs["detector_logits"],
        )
        detector_loss_2 = self.detector_loss_fn(
            homographic_batch["transformed_bins"],
            transformed_outputs["detector_logits"],
        )
        descriptor_loss = self.descriptor_loss_fn(
            homographic_batch["homography_matrices"],
            (outputs["descriptor_logits"], transformed_outputs["descriptor_logits"]),
        )
        loss = (
            detector_loss_1
            + detector_loss_2
            + (self.descriptor_loss_weight * descriptor_loss)
        )

        if self.losses:
            loss = loss + tf.add_n(self.losses)


        return_dict = {
            "loss": loss,
            "detector_loss_1": detector_loss_1,
            "detector_loss_2": detector_loss_2,
            "descriptor_loss": descriptor_loss,
        }
    
        return return_dict
