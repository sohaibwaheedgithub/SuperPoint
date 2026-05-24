from pathlib import Path
import tensorflow as tf


def create_tensorboard_writer(log_dir: Path, suffix=None):
    tb_dir = (log_dir / "tensorboard") if not suffix else (log_dir / "tensorboard" / suffix)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(tb_dir))
