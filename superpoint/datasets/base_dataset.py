from typing import Dict, Tuple, List
import tensorflow as tf
from abc import ABC, abstractmethod

from superpoint.constants import MP_INPUT_SHAPE


class BaseTFRecordDataset(ABC):
    """
    Base class for SuperPoint-style TFRecord datasets.
    Handles:
      - TFRecord reading
      - Example parsing
      - Image decoding & shaping
    """

    def __init__(self):
        self.feature_description: Dict[str, tf.io.FixedLenFeature] = {
            "image": tf.io.FixedLenFeature([], tf.string)
        }


    @abstractmethod
    def _parse_example(self, example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parse a single TFRecord example."""
        pass


    def build_dataset(
        self,
        filenames: List[str],
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        num_parallel_reads = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        
        """Build a tf.data pipeline."""
        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=num_parallel_reads
        )

        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if cache:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.padded_batch(
            batch_size, drop_remainder=True
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    


if __name__ == "__main__":
    from glob import glob
    tfrecord_files = glob("data/tfrecords/synthetic_shapes/train_0/*.tfrecord")[:1]
    base_dataset = BaseTFRecordDataset()
    dataset = base_dataset.build_dataset([tfrecord_files], batch_size=1)
    print(dataset)