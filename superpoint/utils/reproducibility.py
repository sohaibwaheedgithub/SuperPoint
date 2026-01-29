import os
import random
import numpy as np
import tensorflow as tf
from logging import Logger


def set_global_determinism(seed: int, logger: Logger, deterministic: bool = True):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
            logger.info("Successfully enabled Operation determinism via tensorflow API")
        except Exception:
            logger.info("Unable to enable Operation determinism via tensorflow API")
