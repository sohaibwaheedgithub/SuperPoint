import sys
import json
import keras
import random
import tensorflow as tf
from pathlib import Path
from superpoint.constants import MP_INPUT_SHAPE
from superpoint.utils.logging import setup_logger
from superpoint.models.magicpoint import MagicPoint
from superpoint.utils.checkpointing import load_state
from superpoint.utils.tensorboard import create_tensorboard_writer
from superpoint.utils.reproducibility import set_global_determinism
from superpoint.datasets.magicpoint_dataset import MagicPointDataset
from superpoint.configs.magicpoint_config import load_magicpoint_config
from superpoint.callbacks.train_state_checkpoint import TrainingStateCheckpoint
from superpoint.metrics.corner_detection_average_precision import CornerDetectionAveragePrecision
            




def main(config_path: str):
    # 1. Load configuration
    cfg = load_magicpoint_config(config_path)

    # 2. Setup experiment directory
    exp_dir = Path(cfg.logging.root_dir) / cfg.logging.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    
    
    # 3. Setup logging
    logger = setup_logger(exp_dir)
    tb_writer = create_tensorboard_writer(exp_dir)

    # 4. Initialize determinism
    set_global_determinism(
        seed=cfg.runtime.seed,
        logger=logger,
        deterministic=cfg.runtime.deterministic
    )


    logger.info("MagicPoint training initialized")
    logger.info(f"Experiment directory: {exp_dir}")


    logger.info("Building Model")
    
    model = MagicPoint(
        mean=cfg.model.mean,
        variance=cfg.model.variance,
    )
    
    model(tf.zeros((2, *MP_INPUT_SHAPE)))

    logger.info("Model built successfully")

    logger.info("MagicPoint Model Summary")

    model.summary(print_fn=logger.info)
    
    logger.info("Compiling Model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[CornerDetectionAveragePrecision(name="corner_detection_average_precision")]
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

    state_ckpt_cb = TrainingStateCheckpoint(
        ckpt=ckpt,
        last_ckpt_manager=last_ckpt_manager,
        best_ckpt_manager=best_ckpt_manager,
        shard_start=state["shard"],
        tfrecord_start=state["tfrecord"],
        state_path=state_path,
        monitor=cfg.checkpointing.monitor,
        mode=cfg.checkpointing.mode,
    )

    dataset_builder = MagicPointDataset()
    
    logger.info("Building validation dataset")

    valid_tfrecords = [random.choice(tf.io.gfile.glob(
        (Path(cfg.dataset.valid_dir) / "*.tfrecord").as_posix()
    ))]
    assert len(valid_tfrecords) > 0, "No validation data found"

    valid_dataset = dataset_builder.build_dataset(
        valid_tfrecords,
        batch_size=cfg.training.batch_size,
    )
    
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
                callbacks=[state_ckpt_cb],
                verbose=1,
            )
        
        state_ckpt_cb.shard += 1
        state_ckpt_cb.tfrecord_start = 0

if __name__ == "__main__":
    main("configs/magicpoint.yaml")
