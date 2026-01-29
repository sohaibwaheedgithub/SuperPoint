from pathlib import Path
import tensorflow as tf


def create_tensorboard_writer(log_dir: Path):
    tb_dir = log_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(tb_dir))
