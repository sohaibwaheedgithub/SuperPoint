import keras
import tensorflow as tf


class TrainingImageLogger(keras.callbacks.Callback):
    def __init__(self, writer, images, epoch, max_outputs=4):
        super().__init__()
        self._writer = writer
        self._images = tf.convert_to_tensor(images)
        self._epoch = epoch
        self._max_outputs = max_outputs

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self._images, training=False)
        images = tf.clip_by_value(self._images, 0.0, 1.0)
        heatmaps = tf.clip_by_value(outputs["heatmap"], 0.0, 1.0)

        with self._writer.as_default():
            tf.summary.image(
                "ground_truths/images",
                images,
                step=self._epoch,
                max_outputs=self._max_outputs,
            )

            tf.summary.image(
                "predictions/heatmap",
                heatmaps,
                step=self._epoch,
                max_outputs=self._max_outputs,
            )
            self._writer.flush()
