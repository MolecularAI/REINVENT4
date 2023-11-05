"""Dataclasses to hold transformed and aggregated result from components."""

__all__ = ["ComponentResults"]
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


@dataclass
class ComponentResults:
    """Container for the scores, uncertainties and meta data

    At the minimum the scores must be provided.  The order of the score array
    must be the same as the order of SMILES passed to each component.  Failure
    of computation of score must be indicated by NaN. Do not use zero for this!
    scores_properties can be used to pass on meta data on the scores
    uncertainty_type is currently assumed to be the same for all values
    failure_properties can be used to provide details on the failure of a component
    meta_data is a general facility to pass on meta data
    """

    scores: List[np.ndarray]
    scores_properties: Optional[List[Dict]] = None
    uncertainty: Optional[List[np.ndarray]] = None
    uncertainty_type: Optional[str] = None
    uncertainty_properties: Optional[List[Dict]] = None
    failures_properties: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None
