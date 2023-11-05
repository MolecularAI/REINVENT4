from dataclasses import dataclass


@dataclass(frozen=True)
class ModelModeEnum:
    INFERENCE = "inference"
    TRAINING = "training"
