import tensorflow as tf
from superpoint.utils.checkpointing import save_state




class TrainingStateCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        ckpt,
        last_ckpt_manager,
        best_ckpt_manager,
        shard_start,
        tfrecord_start,
        state_path,
        monitor="val_loss",
        mode="min",
    ):
        super().__init__()

        self.ckpt = ckpt
        self.last_ckpt_manager = last_ckpt_manager
        self.best_ckpt_manager = best_ckpt_manager

        self.shard = shard_start
        self.tfrecord_start = tfrecord_start
        self.state_path = state_path

        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

        self.tfrecord_count = 0


    def on_epoch_end(self, epoch, logs=None):
        self.tfrecord_start += 1

        # ---- save training state ----
        #print("Shard inside: ", self.shard)
        save_state(
            self.shard,
            self.tfrecord_start,
            state_path=self.state_path,
        )

        # ---- save "last" checkpoint ----
        if self.last_ckpt_manager:
            self.last_ckpt_manager.save()


        # ---- save "best" checkpoint ----
        if self.best_ckpt_manager:
            current = logs.get(self.monitor)
            if current is None:
                return

            improved = (
                current < self.best if self.mode == "min"
                else current > self.best
            )

            if improved:
                self.best = current
                self.best_ckpt_manager.save()
