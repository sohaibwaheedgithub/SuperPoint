import keras
import tensorflow as tf


class TrainingPRCurveLogger(keras.callbacks.Callback):
    def __init__(self, writer, confidences, epoch):
        super().__init__()
        self._writer = writer
        self._confidences = [float(c) for c in tf.reshape(confidences, [-1])]
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):

        precisions = self.model.cdap_metric.batch_precisions
        recalls = self.model.cdap_metric.batch_recalls

        with self._writer.as_default():
            for i, conf in enumerate(self._confidences):
                tf.summary.scalar(
                    f"pr_curve/precision_at_{conf:.2f}",
                    precisions[i],
                    step=self._epoch,
                )
                tf.summary.scalar(
                    f"pr_curve/recall_at_{conf:.2f}",
                    recalls[i],
                    step=self._epoch,
                )
            self._writer.flush()
