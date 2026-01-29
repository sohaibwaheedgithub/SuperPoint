import tensorflow as tf
from superpoint.utils.checkpointing import save_state


class StateCallback(tf.keras.callbacks.Callback):
    def __init__(self, shard_start, batch_start, state_path):
        self.shard = shard_start
        self.batch = batch_start
        self.state_path = state_path
    
    def on_train_batch_end(self, batch, logs=None):
        save_state(self.shard, self.batch + batch + 1, state_path=self.state_path)