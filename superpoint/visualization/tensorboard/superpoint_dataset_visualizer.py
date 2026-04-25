import tensorflow as tf


class SuperPointDatasetVisualizer:
    """
    TensorBoard visualizer for SuperPoint datasets.
    Logs:
      - Input images
    """

    def __init__(self, log_dir: str):
        self.writer = tf.summary.create_file_writer(log_dir)


    def log_batch(
        self,
        step: int,
        images: tf.Tensor,
        no_images: int,
    ):
        with self.writer.as_default():
            tf.summary.image(
                "dataset/image",
                tf.cast(images, tf.uint8),
                step=step,
                max_outputs=no_images,
            )

        self.writer.flush()


if __name__ == "__main__":
    import argparse
    from glob import glob
    from superpoint.datasets.superpoint_dataset import SuperPointDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/dataset/superpoint")
    parser.add_argument("--no-images", type=int, default=2)
    args = parser.parse_args()

    tfrecord_files = glob("data/tfrecords/ms_coco/train_1/*.tfrecord")[:1]
    superpoint_dataset = SuperPointDataset()
    dataset = superpoint_dataset.build_dataset(tfrecord_files, batch_size=args.no_images)

    visualizer = SuperPointDatasetVisualizer(
        log_dir=args.log_dir
    )

    for step, images in enumerate(dataset.take(1)):
        visualizer.log_batch(
            step=step,
            images=images,
            no_images=args.no_images,
        )
