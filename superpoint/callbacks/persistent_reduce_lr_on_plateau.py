import keras
import numpy as np


class PersistentReduceLROnPlateau(keras.callbacks.Callback):
    def __init__(
        self,
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        cooldown=1,
        min_delta=1e-4,
        mode="min",
        verbose=1,
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best = None
        self.wait = 0
        self.cooldown_counter = 0

    def _get_current_lr(self):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            return float(lr.numpy())
        return float(keras.ops.convert_to_numpy(lr))

    def _set_current_lr(self, new_lr):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(new_lr)
        else:
            self.model.optimizer.learning_rate = new_lr

    def _is_improvement(self, current):
        if self.best is None:
            return True

        if self.mode == "min":
            return current < (self.best - self.min_delta)

        return current > (self.best + self.min_delta)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        current = float(current)

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        old_lr = self._get_current_lr()
        if old_lr <= self.min_lr:
            return

        new_lr = max(old_lr * self.factor, self.min_lr)
        self._set_current_lr(new_lr)

        if self.verbose:
            print(
                f"\nEpoch {epoch + 1}: reducing learning rate "
                f"from {old_lr:.6f} to {new_lr:.6f}"
            )

        self.wait = 0
        self.cooldown_counter = self.cooldown


    def get_state(self):
        return {
            "best": self.best,
            "wait": self.wait,
            "cooldown_counter": self.cooldown_counter,
        }

    def set_state(self, state):
        if not state:
            return

        self.best = state.get("best", self.best)
        self.wait = state.get("wait", 0)
        self.cooldown_counter = state.get("cooldown_counter", 0)

