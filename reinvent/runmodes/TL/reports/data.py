"""Dataclass for reporting """

__all__ = ["TLReportData"]
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TLReportData:
    epoch: int
    model_path: str
    mean_nll: float
    sampled_smilies: Sequence
    sampled_nlls: Sequence
    fingerprints: Sequence
    reference_fingerprints: Sequence
    fraction_valid: float
    fraction_duplicates: float
    internal_diversity: float
    mean_nll_validation: float = None
