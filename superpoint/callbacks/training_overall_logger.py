import keras
import numpy as np
import tensorflow as tf
from superpoint.models.magicpoint import MagicPoint


class TrainingOverAllLogger(keras.callbacks.Callback):
    def __init__(self, writers, images, epoch, max_outputs=4, pred_threshold=0.5):
        super().__init__()
        self._writers = writers
        self._images = tf.convert_to_tensor(images)
        self._epoch = epoch
        self._max_outputs = max_outputs
        self._pred_threshold = pred_threshold

        


    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self._images, training=False)
        
        for block, writer in self._writers.items():
            with writer.as_default():
                block_class = getattr(self.model.encoder, block)
                # Weigths Histogram
                tf.summary.histogram(
                    "conv2d_1/Kernel",
                    block_class.conv2d_1.kernel,
                    step=self._epoch
                )

                tf.summary.histogram(
                    "conv2d_2/Kernel",
                    block_class.conv2d_2.kernel,
                    step=self._epoch
                )

                # Activations Histogram
                tf.summary.histogram(
                    "conv2d_1/Activations",
                    outputs[f"{block}_conv2d_1"],
                    step=self._epoch
                )

                tf.summary.histogram(
                    "conv2d_2/Activations",
                    outputs[f"{block}_conv2d_2"],
                    step=self._epoch
                )

            writer.flush()    