from typing import List, Tuple
import tensorflow as tf

from superpoint.datasets.base_dataset import BaseTFRecordDataset
from superpoint.constants import INPUT_SHAPE


class SuperPointDataset(BaseTFRecordDataset):
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
        self.feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }



    def _parse_example(self, example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Parse the example
        features = tf.io.parse_single_example(example, self.feature_description)
        
        # Decode the image
        image = tf.io.decode_image(features['image_raw'])
        # Set image shape which was lost during parsing
        image = tf.reshape(image, [features["height"], features["width"], features["depth"]])
        
        # Only convert if image is RGB
        image = tf.cond(
            tf.equal(features["depth"], 3),
            lambda: tf.image.rgb_to_grayscale(image),
            lambda: image  # leave as-is if already 1 channel
        )
        
        # Resize
        image = tf.image.resize(image, INPUT_SHAPE[:-1])
        image.set_shape([INPUT_SHAPE[0], INPUT_SHAPE[1], 1])

        return image
    

    def build_dataset(self, filenames, batch_size, shuffle=False, cache=True, num_parallel_reads=None):
        return super().build_dataset(filenames, batch_size, shuffle, cache, num_parallel_reads)


if __name__ == "__main__":
    from glob import glob

    tfrecord_files = glob("data/tfrecords/ms_coco/train_1/*.tfrecord")[:1]
    superpoint_dataset = SuperPointDataset()
    dataset = superpoint_dataset.build_dataset(tfrecord_files, batch_size=2)
    for b in dataset.take(1):
        print(b)
    