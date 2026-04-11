INPUT_SHAPE=[240, 320, 1]  # MP -> Magic Point
MP_BATCH_SIZE=32
SP_BATCH_SIZE=32

import tensorflow as tf
eta = 3
cdap_dtype = tf.float32
detection_confidences = tf.range(0.50, 1, 0.05, dtype=cdap_dtype)