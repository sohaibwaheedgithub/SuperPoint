import tensorflow as tf


def summarize_train_start(writer, vis_batch):
    with writer.as_default():
        tf.summary.image(
            "Visualization Batch",
            vis_batch["image"],
            step=0,
            max_outputs=4,
        )
    writer.flush()

    