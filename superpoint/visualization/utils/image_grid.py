import math
import tensorflow as tf


def images_to_grid(images, repeat=False, scale=None):
    if repeat:
        images = tf.repeat(images, 10 if not scale else scale, axis=1)
        images = tf.repeat(images, 10 if not scale else scale, axis=2)
    
    N, H, W, _ = tf.shape(images)

    num_cols = math.ceil(math.sqrt(N))
    num_rows = math.ceil(N / num_cols)
    border = 2

    # Add white border around each image
    imgs = tf.pad(
        images,
        paddings=[[0, 0], [border, border], [border, border], [0, 0]],
        constant_values=1.0
    )

    H_b = H + 2 * border
    W_b = W + 2 * border

    # Reshape into grid
    grid = tf.reshape(
        imgs,
        [num_rows, num_cols, H_b, W_b, 1]
    )

    # Rearrange dimensions
    grid = tf.transpose(grid, [0, 2, 1, 3, 4])

    # Merge into one large image
    grid = tf.reshape(
        grid,
        [1, num_rows * H_b, num_cols * W_b, 1]
    )

    return grid