"""A dataclass to hold results from a scoring component calculation."""

__all__ = ["TransformResults", "ScoreResults"]
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from reinvent_plugins.components.component_results import ComponentResults


@dataclass(frozen=True)
class TransformResults:
    component_type: str  # internal component type
    component_names: List[str]  # user supplied name for the component score
    transform_types: List[str]
    transformed_scores: List[np.ndarray]
    component_result: ComponentResults
    weight: float = 1.0


@dataclass
class ScoreResults:
    smilies: List[str]
    total_scores: np.ndarray
    completed_components: List[TransformResults]
