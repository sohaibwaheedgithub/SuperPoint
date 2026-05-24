import keras
import numpy as np
import tensorflow as tf
from superpoint.models.magicpoint import MagicPoint


class TrainingOverAllLogger(keras.callbacks.Callback):
    def __init__(self, writer, images, epoch, max_outputs=4, pred_threshold=0.5):
        super().__init__()
        self._writer = writer
        self._images = tf.convert_to_tensor(images)
        self._epoch = epoch
        self._max_outputs = max_outputs
        self._pred_threshold = pred_threshold

        


    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self._images, training=False)

        with self._writer.as_default():
            
            # Weigths Histogram
            tf.summary.histogram(
                "SEConvBlock_1/conv2d_1/Kernel",
                self.model.encoder.SEConvBlock_1.conv2d_1.kernel,
                step=self._epoch
            )

            # Activations Histogram
            tf.summary.histogram(
                "SEConvBlock_1/conv2d_1/Activations",
                outputs["SEConvBlock_1_conv2d_1"],
                step=self._epoch
            )

        self._writer.flush()


    