import keras
import tensorflow as tf


class TensorboardLossLogger(keras.callbacks.Callback):
    def __init__(self, writer):
        super().__init__()
        self._writer = writer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if loss is None and val_loss is None:
            return
        with self._writer.as_default():
            if loss is not None:
                tf.summary.scalar("loss", loss, step=epoch)
            if val_loss is not None:
                tf.summary.scalar("val_loss", val_loss, step=epoch)
            self._writer.flush()
