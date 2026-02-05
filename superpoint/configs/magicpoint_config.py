from dataclasses import dataclass
import yaml


@dataclass
class DatasetConfig:
    train_dir: str
    valid_dir: str
    total_shards: int


@dataclass
class TrainingConfig:
    batch_size: int
    steps_per_epoch: int
    learning_rate: float


@dataclass
class ModelConfig:
    input_channels: int
    mean: float
    variance: float


@dataclass
class RuntimeConfig:
    seed: int
    deterministic: bool


@dataclass
class LoggingConfig:
    root_dir: str
    experiment_name: str


@dataclass
class CheckpointConfig:
    save_last: bool
    last_name: str
    save_best: bool
    best_name: str
    monitor: str
    mode: str
    state_file: str


@dataclass
class MagicPointConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    checkpointing: CheckpointConfig



def load_magicpoint_config(path: str) -> MagicPointConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return MagicPointConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        logging=LoggingConfig(**raw["logging"]),
        checkpointing=CheckpointConfig(**raw["checkpointing"])
    )
