import keras
import tensorflow as tf
from superpoint.models.encoder import SharedEncoder
from superpoint.models.decoder import Decoder
from superpoint.models.post_processor import DetectorPostProcessor





class MagicPoint(keras.Model):
    def __init__(self, mean, variance):
        super().__init__()

        self.encoder = SharedEncoder()
        self.decoder = Decoder(65)
        self.post = DetectorPostProcessor()

        self.mean = tf.constant(mean, dtype=tf.float32)
        self.variance = tf.constant(variance, dtype=tf.float32)


    def call(self, inputs, training=False):
        x = (inputs - self.mean) / tf.sqrt(self.variance)
        x = self.encoder(x, training=training)
        logits = self.decoder(x, training=training)
        heatmap = self.post(logits)

        return {
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

        return {m.name: m.result() for m in self.metrics}



    def test_step(self, data):

        outputs = self(data["image"], training=True)
        self.compute_loss(
            y=data["bins"],
            y_pred=outputs["bins"],
            sample_weight=data["sample_weights"],
        )

        return {m.name: m.result() for m in self.metrics}