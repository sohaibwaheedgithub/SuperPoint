import sys
import json
import keras
import random
import tensorflow as tf
from pathlib import Path
from superpoint.constants import INPUT_SHAPE
from superpoint.utils.logging import setup_logger
from superpoint.models.superpoint import SuperPoint
from superpoint.utils.checkpointing import load_state
from superpoint.utils.tensorboard import create_tensorboard_writer
from superpoint.utils.reproducibility import set_global_determinism
from superpoint.datasets.superpoint_dataset import SuperPointDataset
from superpoint.configs.superpoint_config import load_superpoint_config
from superpoint.callbacks.fit_logger import FitLogger
from superpoint.callbacks.training_image_logger import TrainingImageLogger
from superpoint.callbacks.training_histogram_logger import TrainingHistogramLogger
from superpoint.callbacks.training_pr_curve_logger import TrainingPRCurveLogger
from superpoint.callbacks.training_scalars_logger import TrainingScalarsLogger
from superpoint.callbacks.train_state_checkpoint import TrainingStateCheckpoint
from superpoint.losses.descriptor_loss import DescriptorLoss
from superpoint.training.modules.homographic_adaptation import HomographicAdapter
            

def main(config_path: str):
    # 1. Load configuration
    cfg = load_superpoint_config(config_path)

    # 2. Setup experiment directory
    exp_dir = Path(cfg.logging.root_dir) / cfg.logging.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    
    # 3. Setup logging
    logger = setup_logger(exp_dir)
    tb_writer = create_tensorboard_writer(exp_dir)
    # Route TensorFlow logs to the same handlers as our logger
    tf_logger = tf.get_logger()
    tf_logger.setLevel(logger.level)
    for handler in logger.handlers:
        if handler not in tf_logger.handlers:
            tf_logger.addHandler(handler)

    # 4. Initialize determinism
    set_global_determinism(
        seed=cfg.runtime.seed,
        logger=logger,
        deterministic=cfg.runtime.deterministic
    )


    logger.info("SuperPoint training initialized")
    logger.info(f"Experiment directory: {exp_dir}")
    
    logger.info("Initializing Homographic Adaptor")
    homograhic_adpator = HomographicAdapter(n_homographies=cfg.training.n_homographies)
    logger.info("Successfully initialized Homographic Adaptor")

    logger.info("Building Model")
    
    model = SuperPoint(
        mean=cfg.model.mean,
        variance=cfg.model.variance,
        homographic_adapter=homograhic_adpator
    )
    
    model(tf.zeros((cfg.training.batch_size, *INPUT_SHAPE)))

    logger.info("Model built successfully")

    logger.info("SuperPoint Model Summary")

    model.summary(print_fn=logger.info)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss = {
            "interestPointDecoderOutput": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "descriptorOutput": DescriptorLoss(positive_margin=1.0, negative_margin=0.2, delta=250.0)
        },
        jit_compile=cfg.training.jit_compile
    )

    logger.info("Compilation Completed")


    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state_path = ckpt_dir / cfg.checkpointing.state_file
    state = load_state(state_path)
    logger.info(f"Training State: \n---------------\n{json.dumps(state, indent=4)}")
    
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=model.optimizer,
    )
    
    last_ckpt_manager, best_ckpt_manager = None, None

    if cfg.checkpointing.save_last:
        last_ckpt_manager = tf.train.CheckpointManager(
            ckpt, ckpt_dir / "last", max_to_keep=1
        )

    if cfg.checkpointing.save_best:
        best_ckpt_manager = tf.train.CheckpointManager(
            ckpt, ckpt_dir / "best", max_to_keep=1
        )

    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)

    latest_checkpoint = (
        last_ckpt_manager.latest_checkpoint if last_ckpt_manager else None
    )
    if latest_checkpoint:
        ckpt.restore(latest_checkpoint).expect_partial()
        logger.info(f"Restored training checkpoint: {latest_checkpoint}")
    else:
        logger.info("No previous training checkpoint found. Starting fresh.")

    
    state_ckpt_cb = TrainingStateCheckpoint(
        ckpt=ckpt,
        last_ckpt_manager=last_ckpt_manager,
        best_ckpt_manager=best_ckpt_manager,
        shard_start=state["shard"],
        tfrecord_start=state["tfrecord"],
        epoch_start=state["epoch"],
        state_path=state_path,
        monitor=cfg.checkpointing.monitor,
        mode=cfg.checkpointing.mode,
    )

    dataset_builder = SuperPointDataset()
    
    logger.info("Building validation dataset")

    valid_tfrecords = [random.choice(tf.io.gfile.glob(
        (Path(cfg.dataset.valid_dir) / "*.tfrecord").as_posix()
    ))]
    assert len(valid_tfrecords) > 0, "No validation data found"

    valid_dataset = dataset_builder.build_dataset(
        valid_tfrecords,
        batch_size=cfg.training.batch_size,
        cache=True
    )
    vis_batch = next(iter(valid_dataset.take(1)))
    with tb_writer.as_default():
        tf.summary.image(
            "visuals/images",
            tf.cast(vis_batch, tf.uint8),
            step=0,
            max_outputs=4,
        )
        tb_writer.flush()
    
    logger.info("Validation Dataset Built")


    for shard_idx in range(state_ckpt_cb.shard, cfg.dataset.total_shards+1):
        
        train_tfrecords = sorted(tf.io.gfile.glob(
            (Path(f"{cfg.dataset.train_dir}_{shard_idx}") / "*.tfrecord").as_posix()
        ))[state_ckpt_cb.tfrecord_start:]
        
        logger.info(f"Continuing training from tfrecord: {Path(train_tfrecords[0]).name}")
        
        for train_tfrecord in train_tfrecords:
            train_dataset = dataset_builder.build_dataset(
                [train_tfrecord],
                batch_size=cfg.training.batch_size
            )

            model.fit(
                train_dataset,
                epochs=1,
                steps_per_epoch=cfg.training.steps_per_epoch,
                validation_data=valid_dataset,
                validation_steps=450,
                verbose=1,
            )
            
            
        state_ckpt_cb.advance_to_next_shard()

    

if __name__ == "__main__":
    main("configs/superpoint.yaml")