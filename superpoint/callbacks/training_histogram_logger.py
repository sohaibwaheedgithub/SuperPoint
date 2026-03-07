import keras
import tensorflow as tf


class TrainingHistogramLogger(keras.callbacks.Callback):
    def __init__(self, writer, images, epoch):
        super().__init__()
        self._writer = writer
        self._images = tf.convert_to_tensor(images)
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self._images, training=False)
        with self._writer.as_default():
            for var in self.model.trainable_variables:
                var_path = getattr(var, "path", var.name)
                var_path = var_path.replace("magic_point/", "")
                tag = f"histograms/{var_path}"
                tf.summary.histogram(tag, var, step=self._epoch)

            tf.summary.histogram(
                "activations/shared_encoder",
                outputs["encoder_features"],
                step=self._epoch,
            )
            tf.summary.histogram(
                "activations/decoder_logits",
                outputs["bins"],
                step=self._epoch,
            )
            tf.summary.histogram(
                "activations/heatmap",
                outputs["heatmap"],
                step=self._epoch,
            )
            self._writer.flush()
