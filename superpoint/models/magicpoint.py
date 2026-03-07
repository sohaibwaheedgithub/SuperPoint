import keras
import tensorflow as tf
from superpoint.models.components.encoder import SharedEncoder
from superpoint.models.components.decoder import Decoder
from superpoint.models.components.post_processor import DetectorPostProcessor
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision





class MagicPoint(keras.Model):
    def __init__(self, mean, variance):
        super().__init__()

        self.encoder = SharedEncoder(name="shared_encoder")
        self.decoder = Decoder(65, name="detector_decoder")
        self.post = DetectorPostProcessor(name="detector_post_processor")
        self.cdap_metric = CornerDetectionAveragePrecision(
            name="corner_detection_average_precision"
        )

        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)

    @property
    def metrics(self):
        return [self.cdap_metric]


    def call(self, inputs, training=False):
        x = (inputs - self.mean) / tf.sqrt(self.variance)
        encoder_features = self.encoder(x, training=training)
        logits = self.decoder(encoder_features, training=training)
        heatmap = self.post(logits)

        return {
            "bins": logits,
            "encoder_features": encoder_features,
            "detector_logits": logits,
            "heatmap": heatmap,
        }



    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            outputs = self(data["image"], training=True)
            loss = self.compute_loss(
                y=data["bins"],
                y_pred=outputs["bins"],
                sample_weight=data["sample_weights"],
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        
        # Update Corner Detection Average Precision Metric
        self.cdap_metric.update_state(data["points"], outputs["heatmap"])
        
        return {
            "loss": loss,
            **self.cdap_metric.result()
        }



    def test_step(self, data):
        
        outputs = self(data["image"], training=True)
        loss = self.compute_loss(
            y=data["bins"],
            y_pred=outputs["bins"],
            sample_weight=data["sample_weights"],
        )
        
        self.cdap_metric.update_state(data["points"], outputs["heatmap"])
        
        return {
            "loss": loss,
            **self.cdap_metric.result()
        }