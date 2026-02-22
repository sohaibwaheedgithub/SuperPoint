import keras
import tensorflow as tf


class TrainingHistogramLogger(keras.callbacks.Callback):
    def __init__(self, writer, epoch):
        super().__init__()
        self._writer = writer
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        with self._writer.as_default():
            for var in self.model.trainable_variables:
                var_path = getattr(var, "path", var.name)
                tag = f"histograms/{var_path}"
                tf.summary.histogram(tag, var, step=self._epoch)
            self._writer.flush()
