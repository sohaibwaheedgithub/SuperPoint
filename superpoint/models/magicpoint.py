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


        valid_grads = []
        encoder_grads, decoder_grads = [], []
        encoder_weights, decoder_weights = [], []
        encoder_kernel_grads, encoder_bias_grads = [], []
        encoder_gamma_grads, encoder_beta_grads = [], []
        decoder_kernel_grads, decoder_bias_grads = [], []
        decoder_gamma_grads, decoder_beta_grads = [], []
        encoder_kernel_weights, encoder_bias_weights = [], []
        encoder_gamma_weights, encoder_beta_weights = [], []
        decoder_kernel_weights, decoder_bias_weights = [], []
        decoder_gamma_weights, decoder_beta_weights = [], []

        for grad, var in zip(grads, self.trainable_variables):
            var_path = getattr(var, "path", var.name)
            is_encoder = "shared_encoder" in var_path
            is_decoder = "decoder" in var_path

            is_kernel = var_path.endswith("/kernel")
            is_bias = var_path.endswith("/bias")
            is_gamma = var_path.endswith("/gamma")
            is_beta = var_path.endswith("/beta")

            if is_encoder:
                encoder_weights.append(var)
                if is_kernel:
                    encoder_kernel_weights.append(var)
                elif is_bias:
                    encoder_bias_weights.append(var)
                elif is_gamma:
                    encoder_gamma_weights.append(var)
                elif is_beta:
                    encoder_beta_weights.append(var)

                if grad is not None:
                    valid_grads.append(grad)
                    encoder_grads.append(grad)
                    if is_kernel:
                        encoder_kernel_grads.append(grad)
                    elif is_bias:
                        encoder_bias_grads.append(grad)
                    elif is_gamma:
                        encoder_gamma_grads.append(grad)
                    elif is_beta:
                        encoder_beta_grads.append(grad)

            if is_decoder:
                decoder_weights.append(var)
                if is_kernel:
                    decoder_kernel_weights.append(var)
                elif is_bias:
                    decoder_bias_weights.append(var)
                elif is_gamma:
                    decoder_gamma_weights.append(var)
                elif is_beta:
                    decoder_beta_weights.append(var)

                if grad is not None:
                    valid_grads.append(grad)
                    decoder_grads.append(grad)
                    if is_kernel:
                        decoder_kernel_grads.append(grad)
                    elif is_bias:
                        decoder_bias_grads.append(grad)
                    elif is_gamma:
                        decoder_gamma_grads.append(grad)
                    elif is_beta:
                        decoder_beta_grads.append(grad)


        global_grad_norm = tf.linalg.global_norm(valid_grads) if valid_grads else tf.constant(0.0, tf.float32)
        encoder_grad_norm = tf.linalg.global_norm(encoder_grads) if encoder_grads else tf.constant(0.0, tf.float32)
        decoder_grad_norm = tf.linalg.global_norm(decoder_grads) if decoder_grads else tf.constant(0.0, tf.float32)
        encoder_weight_norm = tf.linalg.global_norm(encoder_weights) if encoder_weights else tf.constant(0.0, tf.float32)
        decoder_weight_norm = tf.linalg.global_norm(decoder_weights) if decoder_weights else tf.constant(0.0, tf.float32)
        encoder_kernel_grad_norm = tf.linalg.global_norm(encoder_kernel_grads) if encoder_kernel_grads else tf.constant(0.0, tf.float32)
        encoder_bias_grad_norm = tf.linalg.global_norm(encoder_bias_grads) if encoder_bias_grads else tf.constant(0.0, tf.float32)
        encoder_gamma_grad_norm = tf.linalg.global_norm(encoder_gamma_grads) if encoder_gamma_grads else tf.constant(0.0, tf.float32)
        encoder_beta_grad_norm = tf.linalg.global_norm(encoder_beta_grads) if encoder_beta_grads else tf.constant(0.0, tf.float32)
        decoder_kernel_grad_norm = tf.linalg.global_norm(decoder_kernel_grads) if decoder_kernel_grads else tf.constant(0.0, tf.float32)
        decoder_bias_grad_norm = tf.linalg.global_norm(decoder_bias_grads) if decoder_bias_grads else tf.constant(0.0, tf.float32)
        decoder_gamma_grad_norm = tf.linalg.global_norm(decoder_gamma_grads) if decoder_gamma_grads else tf.constant(0.0, tf.float32)
        decoder_beta_grad_norm = tf.linalg.global_norm(decoder_beta_grads) if decoder_beta_grads else tf.constant(0.0, tf.float32)
        encoder_kernel_weight_norm = tf.linalg.global_norm(encoder_kernel_weights) if encoder_kernel_weights else tf.constant(0.0, tf.float32)
        encoder_bias_weight_norm = tf.linalg.global_norm(encoder_bias_weights) if encoder_bias_weights else tf.constant(0.0, tf.float32)
        encoder_gamma_weight_norm = tf.linalg.global_norm(encoder_gamma_weights) if encoder_gamma_weights else tf.constant(0.0, tf.float32)
        encoder_beta_weight_norm = tf.linalg.global_norm(encoder_beta_weights) if encoder_beta_weights else tf.constant(0.0, tf.float32)
        decoder_kernel_weight_norm = tf.linalg.global_norm(decoder_kernel_weights) if decoder_kernel_weights else tf.constant(0.0, tf.float32)
        decoder_bias_weight_norm = tf.linalg.global_norm(decoder_bias_weights) if decoder_bias_weights else tf.constant(0.0, tf.float32)
        decoder_gamma_weight_norm = tf.linalg.global_norm(decoder_gamma_weights) if decoder_gamma_weights else tf.constant(0.0, tf.float32)
        decoder_beta_weight_norm = tf.linalg.global_norm(decoder_beta_weights) if decoder_beta_weights else tf.constant(0.0, tf.float32)

        # Update Corner Detection Average Precision Metric
        self.cdap_metric.update_state(data["points"], outputs["heatmap"])

        return {
            "loss": loss,
            "grads/global_norm": global_grad_norm,
            "grads/encoder_norm": encoder_grad_norm,
            "grads/decoder_norm": decoder_grad_norm,
            "grads/encoder_kernel_norm": encoder_kernel_grad_norm,
            "grads/encoder_bias_norm": encoder_bias_grad_norm,
            "grads/encoder_gamma_norm": encoder_gamma_grad_norm,
            "grads/encoder_beta_norm": encoder_beta_grad_norm,
            "grads/decoder_kernel_norm": decoder_kernel_grad_norm,
            "grads/decoder_bias_norm": decoder_bias_grad_norm,
            "grads/decoder_gamma_norm": decoder_gamma_grad_norm,
            "grads/decoder_beta_norm": decoder_beta_grad_norm,
            "weights/encoder_norm": encoder_weight_norm,
            "weights/decoder_norm": decoder_weight_norm,
            "weights/encoder_kernel_norm": encoder_kernel_weight_norm,
            "weights/encoder_bias_norm": encoder_bias_weight_norm,
            "weights/encoder_gamma_norm": encoder_gamma_weight_norm,
            "weights/encoder_beta_norm": encoder_beta_weight_norm,
            "weights/decoder_kernel_norm": decoder_kernel_weight_norm,
            "weights/decoder_bias_norm": decoder_bias_weight_norm,
            "weights/decoder_gamma_norm": decoder_gamma_weight_norm,
            "weights/decoder_beta_norm": decoder_beta_weight_norm,
            **self.cdap_metric.result()
        }



    def test_step(self, data):

        outputs = self(data["image"], training=True)
        loss = self.compute_loss(
            y=data["bins"],
            y_pred=outputs["bins"],
            sample_weight=data["sample_weights"],
        )

        encoder_weights, decoder_weights = [], []
        encoder_kernel_weights, encoder_bias_weights = [], []
        encoder_gamma_weights, encoder_beta_weights = [], []
        decoder_kernel_weights, decoder_bias_weights = [], []
        decoder_gamma_weights, decoder_beta_weights = [], []

        for var in self.trainable_variables:
            var_path = getattr(var, "path", var.name)
            if var_path.startswith("shared_encoder/"):
                encoder_weights.append(var)
                if var_path.endswith("/kernel"):
                    encoder_kernel_weights.append(var)
                elif var_path.endswith("/bias"):
                    encoder_bias_weights.append(var)
                elif var_path.endswith("/gamma"):
                    encoder_gamma_weights.append(var)
                elif var_path.endswith("/beta"):
                    encoder_beta_weights.append(var)
            elif var_path.startswith("decoder/"):
                decoder_weights.append(var)
                if var_path.endswith("/kernel"):
                    decoder_kernel_weights.append(var)
                elif var_path.endswith("/bias"):
                    decoder_bias_weights.append(var)
                elif var_path.endswith("/gamma"):
                    decoder_gamma_weights.append(var)
                elif var_path.endswith("/beta"):
                    decoder_beta_weights.append(var)

        encoder_weight_norm = tf.linalg.global_norm(encoder_weights) if encoder_weights else tf.constant(0.0, tf.float32)
        decoder_weight_norm = tf.linalg.global_norm(decoder_weights) if decoder_weights else tf.constant(0.0, tf.float32)
        encoder_kernel_weight_norm = tf.linalg.global_norm(encoder_kernel_weights) if encoder_kernel_weights else tf.constant(0.0, tf.float32)
        encoder_bias_weight_norm = tf.linalg.global_norm(encoder_bias_weights) if encoder_bias_weights else tf.constant(0.0, tf.float32)
        encoder_gamma_weight_norm = tf.linalg.global_norm(encoder_gamma_weights) if encoder_gamma_weights else tf.constant(0.0, tf.float32)
        encoder_beta_weight_norm = tf.linalg.global_norm(encoder_beta_weights) if encoder_beta_weights else tf.constant(0.0, tf.float32)
        decoder_kernel_weight_norm = tf.linalg.global_norm(decoder_kernel_weights) if decoder_kernel_weights else tf.constant(0.0, tf.float32)
        decoder_bias_weight_norm = tf.linalg.global_norm(decoder_bias_weights) if decoder_bias_weights else tf.constant(0.0, tf.float32)
        decoder_gamma_weight_norm = tf.linalg.global_norm(decoder_gamma_weights) if decoder_gamma_weights else tf.constant(0.0, tf.float32)
        decoder_beta_weight_norm = tf.linalg.global_norm(decoder_beta_weights) if decoder_beta_weights else tf.constant(0.0, tf.float32)
        
        self.cdap_metric.update_state(data["points"], outputs["heatmap"])

        return {
            "loss": loss,
            "weights/encoder_norm": encoder_weight_norm,
            "weights/decoder_norm": decoder_weight_norm,
            "weights/encoder_kernel_norm": encoder_kernel_weight_norm,
            "weights/encoder_bias_norm": encoder_bias_weight_norm,
            "weights/encoder_gamma_norm": encoder_gamma_weight_norm,
            "weights/encoder_beta_norm": encoder_beta_weight_norm,
            "weights/decoder_kernel_norm": decoder_kernel_weight_norm,
            "weights/decoder_bias_norm": decoder_bias_weight_norm,
            "weights/decoder_gamma_norm": decoder_gamma_weight_norm,
            "weights/decoder_beta_norm": decoder_beta_weight_norm,
            **self.cdap_metric.result()
        }
