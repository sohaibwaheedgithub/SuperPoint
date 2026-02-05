import json
import tensorflow as tf

def load_state(state_path):
    if tf.io.gfile.exists(state_path):
        with tf.io.gfile.GFile(state_path, "r") as f:
            return json.load(f)
    return {"shard": 1, "tfrecord": 0, "epoch": 1}



def save_state(shard, tfrecord, epoch, state_path):
    tf.io.gfile.GFile(state_path, "w").write(
        f'{{"shard": {shard}, "tfrecord": {tfrecord}, "epoch": {epoch}}}'
    )