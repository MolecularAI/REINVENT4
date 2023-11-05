"""Various data classes defining DTOs needed in TL."""

__all__ = ["SampledStatsDTO", "CollectedStatsDTO"]
from dataclasses import dataclass
from typing import List


@dataclass
class SampledStatsDTO:
    nll_input_sampled_target: List[float]
    molecule_smiles: List[str]
    molecule_parts_smiles: List[str]
    valid_fraction: float


@dataclass
class CollectedStatsDTO:
    jsd_binned: float
    jsd_un_binned: float
    nll: List[float]
    training_stats: SampledStatsDTO
    validation_nll: List[float] = None
    validation_stats: SampledStatsDTO = None
