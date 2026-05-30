from pathlib import Path
import tensorflow as tf


def create_tensorboard_writer(log_dir: Path, sub_dir=None):
    tb_dir = (log_dir / "tensorboard") if not sub_dir else (log_dir / "tensorboard" / sub_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(tb_dir))
