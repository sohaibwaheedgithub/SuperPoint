import keras


class FitLogger(keras.callbacks.Callback):
    def __init__(self, logger, epoch):
        super().__init__()
        self._logger = logger
        self.epoch = epoch


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        self._logger.info(f"Epoch {self.epoch}]: {metrics}")