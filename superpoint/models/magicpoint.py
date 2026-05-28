import keras
import tensorflow as tf
from superpoint.models.components.encoder import SharedEncoder
from superpoint.models.components.decoder import Decoder
from superpoint.models.components.post_processor import DetectorPostProcessor
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision





class MagicPoint(keras.Model):
    def __init__(self, mean, variance, writer):
        super().__init__()

        self.encoder = SharedEncoder(name="shared_encoder")
        self.decoder = Decoder(65, name="detector_decoder")
        self.post = DetectorPostProcessor(name="detector_post_processor")

        self.cdap_metric = CornerDetectionAveragePrecision(
            name="corner_detection_average_precision"
        )

        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)

        self._epoch = 0
        self._writer = writer



    def call(self, inputs, training=False):
        x = (inputs - self.mean) / tf.sqrt(self.variance)
        encoder_features = self.encoder(x, training=training)
        #logits = self.decoder(encoder_features, training=training)
        logits = self.decoder(encoder_features["SEConvBlock_4"]["batchNorm_2"], training=training)
        heatmap = self.post(logits)

        return {
            #"encoder_features": encoder_features,
            "SEConvBlock_1_conv2d_1": encoder_features["SEConvBlock_1"]["conv2d_1"],
            "encoder_features": encoder_features["SEConvBlock_4"]["batchNorm_2"],
            "bins": logits,
            "heatmap": heatmap,
        }
    

    def log_gradients(self, step, grads):
        
        kernel = self.encoder.SEConvBlock_1.conv2d_1.kernel
        for var, grad in zip(self.trainable_variables, grads):
            if var is kernel:
                with tf.name_scope(None):
                    tf.summary.histogram(
                        "SEConvBlock_1/conv2d_1/Gradients",
                        grad,
                        step=step * self._epoch
                    )

        return tf.no_op()



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
        
        with self._writer.as_default():
            tf.cond(
                tf.equal(step % 50, 0),
                lambda: self.log_gradients(step, grads),
                lambda: tf.no_op()
            )
        self._writer.flush()
        
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