import os
import sys
from glob import glob
from pathlib import Path
import tensorflow as tf

from superpoint.constants import INPUT_SHAPE
from superpoint.configs.magicpoint_config import load_magicpoint_config
from superpoint.utils.logging import setup_logger



INPUT_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "points": tf.io.FixedLenFeature([], tf.string),
}


def generate_bins(points):
    points = tf.round(points)
    # To prepare all possible set of coordinates of points in the image
    x = range(0, INPUT_SHAPE[0])
    y = range(0, INPUT_SHAPE[1])
    X, Y = tf.meshgrid(x, y, indexing="ij")
    # Shaping it up in this form so that points can be compared using tf.equal
    X, Y = X[..., tf.newaxis], Y[..., tf.newaxis]
    gridsRegion = tf.reshape(tf.cast(tf.concat([X, Y], axis=-1), tf.float32), [-1, 1, 2])
    # Comparing each coordinate position with all ground truth points to get a tensor of shape [total_cooridnates, n_gt_pts, 2]
    # For a point to lie on a pixel, both of it's coordinates should match with pixel's both coordinates i.e [True, True]
    # Then reducing [bool, bool] -> [bool] to get only those pixels where both coordinates match
    binsBooleanMask = tf.reduce_all(tf.equal(gridsRegion, points[tf.newaxis, ...]), axis=-1)
    # Reshaping [total_cooridnates, n_gt_pts] -> [120, 160, n_gt_pts]
    binsBooleanMask = tf.reshape(binsBooleanMask, [INPUT_SHAPE[0], INPUT_SHAPE[1], -1])
    # converting True -> 1 and False -> 0, since amoung all points there exists only one point that lies on a certain
    # pixel, then if we sum all points together we will get 1 for pixels where points lie and 0 for pixels where points
    # doesn't lie
    # Also adding and batch dimension and depth dimension as tf.nn.space_to_depth expects so
    binsBinaryMask = tf.reduce_sum(tf.cast(binsBooleanMask, tf.float32), axis=-1)[tf.newaxis, ..., tf.newaxis]
    # Now extracting patches of size 30 x 40 x 64 from the image by sliding 8 x 8 window (going from space to depth)
    bins = tf.nn.space_to_depth(binsBinaryMask, block_size=8)[0]
    bins = tf.concat([bins, tf.ones_like(bins)[..., -1:]*0.5], axis=-1)
    bins = tf.argmax(bins, axis=-1, output_type=tf.int32)
    return bins


def parse_example(example):
    # Images saved in tfrecords are of resolution 120 x 160, along with points of coordinates within 120 x 160 grid
    # That's why need to rescale image and points
    # image will be rescaled in dataset builder on the run (to save disk space)
    # But rescaled points have to be saved in disk as bins generation is dependant on points
    parsed = tf.io.parse_single_example(example, INPUT_FEATURES)
    image = parsed["image"]                                                  
    points = tf.io.parse_tensor(parsed["points"], out_type=tf.float32) / 2    # Since some records were processed incorrectly, the ones with size around 155..... Kbs, so need to divide their points first by 2
    scale_y = INPUT_SHAPE[0] / 120
    scale_x = INPUT_SHAPE[1] / 160
    points *= [scale_y, scale_x]                                    
    points.set_shape([None, 2])

    return image, points


def serialize_example(image, points, bins):
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.numpy()])
        ),
        "points": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(points).numpy()]
            )
        ),
        "bins": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(bins).numpy()]
            )
        ),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example.SerializeToString()


def process_tfrecord(path: str):
    tmp_path = path + ".tmp"

    writer = tf.io.TFRecordWriter(tmp_path)

    for raw in tf.data.TFRecordDataset([path]):
        image, points = parse_example(raw)

        bins = generate_bins(points)

        serialized = serialize_example(image, points, bins)
        writer.write(serialized)

    writer.close()

    os.replace(tmp_path, path)


if __name__ == "__main__":
    cfg = load_magicpoint_config("configs/magicpoint.yaml")
    exp_dir = Path(cfg.logging.root_dir) / cfg.logging.experiment_name
    logger = setup_logger(exp_dir)

    tfrecord_files = sorted(glob("data/tfrecords/synthetic_shapes/train_3/*.tfrecord"))
    total_files = len(tfrecord_files)

    logger.info("Starting bin precomputation")
    logger.info(f"Writing logs to: {exp_dir / 'train.log'}")

    for file_idx, tfrecord in enumerate(tfrecord_files, start=1):
        tfrecord_path = Path(tfrecord)
        logger.info(
            f"Processing dir: {tfrecord_path.parent} | file {file_idx}/{total_files}: {tfrecord_path.name}"
        )
        process_tfrecord(tfrecord)

    logger.info("Bins added to TFRecords")

