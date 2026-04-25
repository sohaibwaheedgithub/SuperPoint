import tensorflow as tf
from superpoint.utils.checkpointing import save_state




class TrainingStateCheckpoint(tf.keras.callbacks.Callback):
    # def __init__(
    #     self,
    #     ckpt,
    #     last_ckpt_manager,
    #     best_ckpt_manager,
    #     shard_start,
    #     tfrecord_start,
    #     epoch_start,
    #     state_path,
    #     monitor="val_loss",
    #     mode="min",
    # ):

    def __init__(
        self,
        ckpt,
        last_ckpt_manager,
        best_ckpt_manager,
        shard_start,
        tfrecord_start,
        epoch_start,
        state_path,
        monitor="val_loss",
        mode="min",
        scheduler_cb=None,
    ):
        super().__init__()

        self.ckpt = ckpt
        self.last_ckpt_manager = last_ckpt_manager
        self.best_ckpt_manager = best_ckpt_manager

        self.shard = shard_start
        self.tfrecord_start = tfrecord_start
        self.epoch_start = epoch_start
        self.state_path = state_path

        self.monitor = monitor
        self.mode = mode
        self.scheduler_cb = scheduler_cb

        self.best = float("inf") if mode == "min" else -float("inf")


    # def _save_training_state(self):
    #     save_state(
    #         self.shard,
    #         self.tfrecord_start,
    #         self.epoch_start,
    #         state_path=self.state_path,
    #     )
    
    def _save_training_state(self):
        scheduler_state = None
        if self.scheduler_cb is not None:
            scheduler_state = self.scheduler_cb.get_state()

        save_state(
            self.shard,
            self.tfrecord_start,
            self.epoch_start,
            state_path=self.state_path,
            scheduler=scheduler_state,
        )



    def on_epoch_end(self, epoch, logs=None):
        self.tfrecord_start += 1
        self.epoch_start += 1

        # ---- save training state ----
        self._save_training_state()

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


    def advance_to_next_shard(self):
        self.shard += 1
        self.tfrecord_start = 0
        self._save_training_state()
