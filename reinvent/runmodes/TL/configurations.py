"""Configurations for TL."""

from __future__ import annotations

__all__ = [
    "GeneralConfiguration",
    "Mol2MolConfiguration",
    "StepLRConfiguration",
    "NoamoptConfiguration",
]
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reinvent.chemistry.standardization.filter_configuration import (
        FilterConfiguration,
    )
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR, StepLR


@dataclass(frozen=True)
class Configuration:
    input_model_file: str
    output_model_file: str
    smilies: List[str]
    optimizer: Optimizer
    learning_rate_scheduler: None
    learning_rate_config: StepLRConfiguration
    n_cpus: int
    validation_smilies: Optional[str] = None
    save_every_n_epochs: int = 1
    batch_size: int = 128
    sample_batch_size: int = 128
    num_epochs: int = 10
    num_refs: int = 0  # number of reference molecules for similarity
    starting_epoch: int = 1
    shuffle_each_epoch: bool = True
    randomize_all_smiles: bool = False
    internal_diversity: bool = False


@dataclass(frozen=True)
class GeneralConfiguration(Configuration):
    learning_rate_scheduler: StepLR
    clip_gradient_norm: float = 1.0
    standardization_filters: List[FilterConfiguration] = field(default_factory=list)


@dataclass(frozen=True)
class Mol2MolConfiguration(Configuration):
    learning_rate_scheduler: LambdaLR
    max_sequence_length: int = 128
    pairs: dict = field(
        default_factory=lambda: {
            "type": "tanimoto",
            "upper_threshold": 1.0,
            "lower_threshold": 0.7,
            "min_cardinality": 1,
            "max_cardinality": 199,
        }
    )  # FIXME: make a configuration object
    ranking_loss_penalty: bool = False
    reset_optimizer: bool = True
    validation_percentage: float = 0.0
    validation_seed: int = None


@dataclass(frozen=True)
class StepLRConfiguration:
    start: float = 0.0001
    min: float = 0.000001
    gamma: float = 0.95
    step: int = 1


@dataclass(frozen=True)
class NoamoptConfiguration:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    factor: float = 1.0
    warmup: float = 4000
