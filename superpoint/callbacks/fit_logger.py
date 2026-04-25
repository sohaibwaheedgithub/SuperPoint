import keras


class FitLogger(keras.callbacks.Callback):
    def __init__(self, logger, epoch):
        super().__init__()
        self._logger = logger
        self.epoch = epoch

    def _get_current_lr(self):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            return float(lr.numpy())
        return float(keras.ops.convert_to_numpy(lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        current_lr = self._get_current_lr()
        if metrics:
            metrics = f"{metrics} learning_rate={current_lr:.8f}"
        else:
            metrics = f"learning_rate={current_lr:.8f}"
        self._logger.info(f"Epoch {self.epoch}]: {metrics}")
