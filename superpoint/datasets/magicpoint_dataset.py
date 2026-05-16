from typing import List, Tuple
import tensorflow as tf

from superpoint.datasets.base_dataset import BaseTFRecordDataset
from superpoint.constants import INPUT_SHAPE


class MagicPointDataset(BaseTFRecordDataset):
    """
    Dataset for MagicPoint training.
    Loads:
      - image
      - points
      - bins
    """

    def __init__(self):
        super().__init__()

        # Extend base feature description
        self.feature_description.update({
            "points": tf.io.FixedLenFeature([], tf.string),
            "bins": tf.io.FixedLenFeature([], tf.string)
        })


    def _compute_sample_weights(self, bins):
        bins = tf.cast(bins != 64, tf.float32)   # 1 = point, 0 = non-point
        n_points = tf.reduce_sum(bins, axis=[0, 1], keepdims=True)
        n_points = tf.maximum(n_points, 1.0)
        total = tf.cast(tf.size(bins), tf.float32)
        n_non_points = total - n_points
        pos_weight = n_non_points / n_points
        sample_weights = bins * pos_weight + (1.0 - bins)
        return sample_weights


    def _parse_example(self, example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        parsed = tf.io.parse_single_example(example, self.feature_description)

        image = tf.io.decode_png(parsed["image"], channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, INPUT_SHAPE[:2])
        image.set_shape(INPUT_SHAPE)
     
        points = tf.io.parse_tensor(parsed["points"], out_type=tf.float32)
        points.set_shape([None, 2])

        bins = tf.io.parse_tensor(
            parsed["bins"], out_type=tf.int32
        )
        bins.set_shape([INPUT_SHAPE[0]//8, INPUT_SHAPE[1]//8])

        # Compute sample weights of each point
        sample_weights = self._compute_sample_weights(bins)

        return {
            "image": image, 
            "points": points, 
            "bins": bins, 
            "sample_weights": sample_weights
        }
    

    def build_dataset(self, filenames, batch_size, shuffle=False, cache=True, num_parallel_reads=None):
        # Force stop shuffling and num_parallel_reads for magicpoint dataset
        shuffle=False
        num_parallel_reads=None
        return super().build_dataset(filenames, batch_size, shuffle, cache, num_parallel_reads)


if __name__ == "__main__":
    from glob import glob

    tfrecord_files = glob("data/tfrecords/synthetic_shapes/train_1/*.tfrecord")[:1]
    magicpoint_dataset = MagicPointDataset()
    dataset = magicpoint_dataset.build_dataset([tfrecord_files], batch_size=32)

    for i in dataset.take(1):
        print(i["image"])
        break
