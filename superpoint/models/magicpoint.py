import keras
import tensorflow as tf
from superpoint.models.components.encoder import SharedEncoder
from superpoint.models.components.decoder import Decoder
from superpoint.models.components.post_processor import DetectorPostProcessor
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision





class MagicPoint(keras.Model):
    def __init__(self, mean, variance, writers: dict):
        super().__init__()

        self.encoder = SharedEncoder(name="shared_encoder")
        self.decoder = Decoder(65, name="detector_decoder")
        self.post = DetectorPostProcessor(name="detector_post_processor")

        self.cdap_metric = CornerDetectionAveragePrecision(
            name="corner_detection_average_precision"
        )

        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)

        self._writers = writers



    def call(self, inputs, training=False):
        x = (inputs - self.mean) / tf.sqrt(self.variance)
        encoder_features = self.encoder(x, training=training)
        #logits = self.decoder(encoder_features, training=training)
        logits = self.decoder(encoder_features["SEConvBlock_4"]["batchNorm_2"], training=training)
        heatmap = self.post(logits)

        return {
            "SEConvBlock_1": encoder_features["SEConvBlock_1"],
            "maxPool_1": encoder_features["maxPool_1"],
            "SEConvBlock_2": encoder_features["SEConvBlock_2"],
            "maxPool_2": encoder_features["maxPool_2"],
            "SEConvBlock_3": encoder_features["SEConvBlock_3"],
            "maxPool_3": encoder_features["maxPool_3"],
            "SEConvBlock_4": encoder_features["SEConvBlock_4"],
            "encoder_features": encoder_features["SEConvBlock_4"]["batchNorm_2"],
            "bins": logits,
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
        
        step = self.optimizer.iterations

        grads_by_var = {
            id(var): grad
            for var, grad in zip(self.trainable_variables, grads)
            if grad is not None
        }

        for block, writer in self._writers.items():
            if hasattr(self.encoder, block):
                block_object = getattr(self.encoder, block)

                with writer.as_default():
                    for conv_name in ("conv2d_1", "conv2d_2"):
                        conv_layer = getattr(block_object, conv_name)
                        grad = grads_by_var.get(id(conv_layer.kernel))

                        if grad is not None:
                            update_norm = tf.norm(self.optimizer.learning_rate * grad)
                            weight_norm = tf.norm(conv_layer.kernel)

                            ratio = update_norm / (weight_norm + 1e-8)
                            
                            with tf.summary.record_if(True):
                                tf.summary.scalar(
                                    f"{conv_name}/UpdateWeightRatio",
                                    ratio,
                                    step=step
                                )

                            with tf.summary.record_if(tf.equal(step % 50, 0)):
                                tf.summary.histogram(
                                    f"{conv_name}/Gradients",
                                    grad,
                                    step=step,
                                )

                writer.flush()
        

        return_dict = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                return_dict[metric.name] = metric.result()

        self.cdap_metric.update_state(data["points"], outputs["heatmap"])
        return_dict.update(self.cdap_metric.result())

        self.latest_gradients = grads

        return return_dict



    def test_step(self, data):
        
        outputs = self(data["image"], training=False)
        loss = self.compute_loss(
            y=data["bins"],
            y_pred=outputs["bins"],
            sample_weight=data["sample_weights"],
        )

        return_dict = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                return_dict[metric.name] = metric.result()

        self.cdap_metric.update_state(data["points"], outputs["heatmap"])
        return_dict.update(self.cdap_metric.result())

        return return_dict



if __name__ == "__main__":
    from superpoint.constants import INPUT_SHAPE
    model = MagicPoint(
        mean=0.5,
        variance=0.25,
    )
    
    model(tf.zeros((1, *INPUT_SHAPE)))

    print(model.encoder.SEConvBlock_1.conv2d_1.kernel.shape)
