import keras
import tensorflow as tf
from superpoint.models.encoder import SharedEncoder
from superpoint.models.decoder import Decoder
from superpoint.models.post_processor import DetectorPostProcessor
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision





class MagicPoint(keras.Model):
    def __init__(self, mean, variance):
        super().__init__()

        self.encoder = SharedEncoder(name="shared_encoder")
        self.decoder = Decoder(65, name="decoder")
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
        x = self.encoder(x, training=training)
        logits = self.decoder(x, training=training)
        heatmap = self.post(logits)

        return {
            "bins": logits,
            "heatmap": heatmap,
        }

    def _safe_global_norm(self, tensors):
        valid_tensors = [t for t in tensors if t is not None]
        if not valid_tensors:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.linalg.global_norm(valid_tensors)



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


        encoder_grads, decoder_grads = [], []
        encoder_weights, decoder_weights = [], []

        for grad, var in zip(grads, self.trainable_variables):
            tf.print("Varname: ", var.name)
            if "shared_encoder" in var.name:
                encoder_weights.append(var)
                if grad is not None:
                    encoder_grads.append(grad)
            elif "decoder" in var.name:
                decoder_weights.append(var)
                if grad is not None:
                    decoder_grads.append(grad)


        self.cdap_metric.update_state(data["points"], outputs["heatmap"])

        valid_grads = [g for g in grads if g is not None]

        global_grad_norm = tf.linalg.global_norm(valid_grads) if valid_grads else tf.constant(0.0, tf.float32)
        encoder_grad_norm = tf.linalg.global_norm(encoder_grads) if encoder_grads else tf.constant(0.0, tf.float32)
        decoder_grad_norm = tf.linalg.global_norm(decoder_grads) if decoder_grads else tf.constant(0.0, tf.float32)
        encoder_weight_norm = tf.linalg.global_norm(encoder_weights) if encoder_weights else tf.constant(0.0, tf.float32)
        decoder_weight_norm = tf.linalg.global_norm(decoder_weights) if decoder_weights else tf.constant(0.0, tf.float32)

        return {
            "loss": loss,
            "grads/global_norm": global_grad_norm,
            "grads/encoder_norm": encoder_grad_norm,
            "grads/decoder_norm": decoder_grad_norm,
            "weights/encoder_norm": encoder_weight_norm,
            "weights/decoder_norm": decoder_weight_norm,
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

        encoder_weights, decoder_weights = [], []

        for var in self.trainable_variables:
            if "shared_encoder" in var.name:
                encoder_weights.append(var)
            elif "decoder" in var.name:
                decoder_weights.append(var)

        encoder_weight_norm = tf.linalg.global_norm(encoder_weights) if encoder_weights else tf.constant(0.0, tf.float32)
        decoder_weight_norm = tf.linalg.global_norm(decoder_weights) if decoder_weights else tf.constant(0.0, tf.float32)


        return {
            "loss": loss,
            "weights/encoder_norm": encoder_weight_norm,
            "weights/decoder_norm": decoder_weight_norm,
            **self.cdap_metric.result()
        }
