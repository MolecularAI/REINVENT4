"""Dataclass for reporting """

from __future__ import annotations

__all__ = ["RLReportData"]
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from reinvent.scoring import ScoreResults
    from reinvent.models.model_factory.sample_batch import SampleBatch


@dataclass
class RLReportData:
    step: int
    stage: int
    smilies: list
    isim: Optional[float]
    scaffolds: list
    sampled: SampleBatch
    score_results: ScoreResults
    prior_mean_nll: float
    agent_mean_nll: float
    augmented_mean_nll: float
    prior_nll: float
    agent_nll: float
    augmented_nll: float
    loss: float
    fraction_valid_smiles: float
    fraction_duplicate_smiles: float
    df_memory_smilies: int
    bucket_max_size: int
    num_full_buckets: int
    num_total_buckets: int
    mean_score: float
    model_type: str
    start_time: float
    n_steps: int
    mask_idx: np.ndarray
