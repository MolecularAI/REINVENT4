"""Compute scores with test score:)"""

from __future__ import annotations

__all__ = ["TestScore"]
import pickle
from typing import List
import logging
import json

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    target: List[str]


@add_tag("__component")
class TestScore:
    def __init__(self, params: Parameters):
        self.targets = params.target

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.ndarray:
        scores = []

        for smi in smilies:
            score = 0.0
            for target in self.targets:
                for char in smi:
                    if char == target:
                        score += 1.0
            if score == 0.0:
                score = "NaN"
            scores.append([score])  

        scores = np.array(scores)  # shape (N,1)

        print("SCORES shape:", scores.shape)

        return ComponentResults(scores)

