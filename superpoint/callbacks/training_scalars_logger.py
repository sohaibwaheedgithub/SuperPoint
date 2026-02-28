import keras
import tensorflow as tf


class TrainingScalarsLogger(keras.callbacks.Callback):
    def __init__(self, writer, epoch):
        super().__init__()
        self._writer = writer
        self._epoch = epoch 

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        with self._writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=self._epoch)

            lr_value = tf.convert_to_tensor(self.model.optimizer.learning_rate)
            tf.summary.scalar("learning_rate", lr_value, step=self._epoch)

            self._writer.flush()
