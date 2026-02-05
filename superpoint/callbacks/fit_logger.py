import keras


class FitLogger(keras.callbacks.Callback):
    def __init__(self, logger, context=None):
        super().__init__()
        self._logger = logger
        self._context = context

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        if self._context:
            self._logger.info(f"Epoch {epoch + 1} [{self._context}]: {metrics}")
        else:
            self._logger.info(f"Epoch {epoch + 1}: {metrics}")
