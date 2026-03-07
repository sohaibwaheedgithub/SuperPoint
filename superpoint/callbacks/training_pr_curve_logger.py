import keras
import tensorflow as tf


class TrainingPRCurveLogger(keras.callbacks.Callback):
    def __init__(self, writer, confidences, epoch, cdap_metric):
        super().__init__()
        self._writer = writer
        self._confidences = [float(c) for c in tf.reshape(confidences, [-1])]
        self._epoch = epoch
        self._cdap_metric = cdap_metric

    def on_epoch_end(self, epoch, logs=None):

        precisions = self._cdap_metric.batch_precisions
        recalls = self._cdap_metric.batch_recalls
        sort_idx = tf.argsort(recalls)
        recalls_sorted = tf.gather(recalls, sort_idx)
        precisions_sorted = tf.gather(precisions, sort_idx)
        pr_auc = tf.reduce_sum(
            (recalls_sorted[1:] - recalls_sorted[:-1])
            * (precisions_sorted[1:] + precisions_sorted[:-1])
            * 0.5
        )

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
            tf.summary.scalar("pr_curve/auc", pr_auc, step=self._epoch)
            self._writer.flush()
