import cv2
import numpy as np
import tensorflow as tf


class MagicPointDatasetVisualizer:
    """
    TensorBoard visualizer for MagicPoint datasets.
    Logs:
      - Input images
      - Detector heatmaps (bins projected to pixel space)
    """

    def __init__(self, log_dir: str, grid_size: int = 8):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.grid_size = grid_size


    def _bins_to_heatmap(self, bins: tf.Tensor) -> tf.Tensor:
        """
        Convert bins (H/8, W/8) -> pixel heatmap (H, W, 1)
        """
        # bins: (B, H/8, W/8)
        b, h, w = tf.unstack(tf.shape(bins))
        depth = self.grid_size * self.grid_size

        # one-hot (exclude dustbin assumed already)
        one_hot = tf.one_hot(bins, depth=depth)

        # depth_to_space expects NHWC
        one_hot = tf.reshape(one_hot, [b, h, w, depth])
        heatmap = tf.nn.depth_to_space(one_hot, self.grid_size)

        # (B, H, W, 1)
        heatmap = tf.reduce_max(heatmap, axis=-1, keepdims=True)
        return heatmap

    
    def _sample_weights_to_heatmap(self, sample_weights: tf.Tensor) -> tf.Tensor:
        """
        Convert sample weights (B, H/8, W/8) -> pixel heatmap (B, H, W, 1)
        """
        depth = self.grid_size * self.grid_size

        # replicate weights across sub-pixels
        sample_weights = tf.expand_dims(sample_weights, axis=-1)          # (B, H/8, W/8, 1)
        sample_weights = tf.tile(sample_weights, [1, 1, 1, depth])         # (B, H/8, W/8, 64)

        heatmap = tf.nn.depth_to_space(sample_weights, self.grid_size)

        return heatmap



    def draw_points(
        self,
        image: tf.Tensor,
        points: tf.Tensor,
        color=(0, 255, 0),
        radius=3,
    ):
        """
        image: (H, W, 1) or (H, W, 3), float [0,1]
        points: (N, 2) in (y, x)
        """
        img = image.numpy().astype(np.uint8)
        #img = (img * 255).astype(np.uint8)

        if img.ndim == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        
        for y, x in points.numpy().astype(int):
            cv2.circle(img, (x, y), radius, color, -1)

        return tf.convert_to_tensor(img / 255.0, tf.float32)
    

    def draw_bins(
        self,
        image: tf.Tensor,
        heatmap: tf.Tensor,
        color=(0, 0, 255),
        threshold=0.5,
        radius=3,
    ):
        img = image.numpy().astype(np.uint8)
        if img.ndim == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = heatmap.numpy()[..., 0] > threshold
        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):
            cv2.circle(img, (x, y), radius, color, -1)

        return tf.convert_to_tensor(img / 255.0, tf.float32)
    


    def draw_sample_weights(
        self,
        image: tf.Tensor,
        heatmap: tf.Tensor,
        color=(255, 0, 0),
        threshold=1,
        radius=1,
    ):
        img = image.numpy().astype(np.uint8)
        if img.ndim == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = heatmap.numpy()[..., 0] > threshold
        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):
            cv2.circle(img, (x, y), radius, color, -1)

        return tf.convert_to_tensor(img / 255.0, tf.float32)



    def log_batch(
        self,
        step: int,
        images: tf.Tensor,
        points: tf.Tensor,
        bins: tf.Tensor,
        sample_weights: tf.Tensor, 
        no_images: int
    ):

        bins_heatmaps = self._bins_to_heatmap(bins)
        sample_weights_heatmaps = self._sample_weights_to_heatmap(sample_weights)


        with self.writer.as_default():
            tf.summary.image(
                "dataset/image",
                tf.cast(images, tf.uint8),
                step=step,
                max_outputs=no_images,
            )

            for i in range(images.shape[0]):
                
                gt_overlay = self.draw_points(
                    images[i],
                    points[i],
                    color=(0, 255, 0),
                    radius=3,
                )

                bin_overlay = self.draw_bins(
                    images[i],
                    bins_heatmaps[i],
                    color=(0, 0, 255),
                )

                sample_weights_overlay = self.draw_sample_weights(
                    images[i],
                    sample_weights_heatmaps[i],
                    color=(255, 0, 0),
                )


                tf.summary.image(
                    f"dataset/image_with_gt_points/{i}",
                    gt_overlay[tf.newaxis, ...],
                    step=step,
                )

                tf.summary.image(
                    f"dataset/image_with_bins/{i}",
                    bin_overlay[tf.newaxis, ...],
                    step=step,
                )

                tf.summary.image(
                    f"dataset/image_with_sample_weights/{i}",
                    sample_weights_overlay[tf.newaxis, ...],
                    step=step
                )

                tf.summary.histogram(
                    "dataset/raw_data_distribution",
                    images[i],
                    step=int(i),
                )

                tf.summary.histogram(
                    "dataset/standardized_data_distribution",
                    (images[i] - 0.50) / 0.50,
                    step=int(i),
                )

        self.writer.flush()


if __name__ == "__main__":
    import argparse
    from glob import glob
    from superpoint.datasets.magicpoint_dataset import MagicPointDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/dataset/magicpoint")
    parser.add_argument("--no-images", type=int, default=2)
    args = parser.parse_args()

    tfrecord_files = glob("data/tfrecords/synthetic_shapes/train_1/*.tfrecord")[:1]
    magicpoint_dataset = MagicPointDataset()
    dataset = magicpoint_dataset.build_dataset([tfrecord_files], batch_size=args.no_images)

    visualizer = MagicPointDatasetVisualizer(
        log_dir=args.log_dir
    )

    for step, data in enumerate(dataset.take(1)):
        visualizer.log_batch(step, data["image"], data["points"], data["bins"], data["sample_weights"], no_images=args.no_images)
