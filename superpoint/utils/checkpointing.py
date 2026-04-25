import json
import tensorflow as tf


def load_state(state_path):
    if tf.io.gfile.exists(state_path):
        with tf.io.gfile.GFile(state_path, "r") as f:
            state = json.load(f)

        state.setdefault("shard", 1)
        state.setdefault("tfrecord", 0)
        state.setdefault("epoch", 1)
        state.setdefault("scheduler", {})
        return state

    return {
        "shard": 1,
        "tfrecord": 0,
        "epoch": 1,
        "scheduler": {},
    }


def save_state(shard, tfrecord, epoch, state_path, scheduler=None):
    state = {
        "shard": shard,
        "tfrecord": tfrecord,
        "epoch": epoch,
        "scheduler": scheduler or {},
    }

    with tf.io.gfile.GFile(state_path, "w") as f:
        json.dump(state, f)
