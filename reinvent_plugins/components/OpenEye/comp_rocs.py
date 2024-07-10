"""OpenEye ROCS
Example toml config
[component.rocssimilarity]
[[component.rocssimilarity.endpoint]]
name = "s58 rocs sq"
weight = 1
params.color_weight = 0.5
params.shape_weight = 0.5
params.similarity_measure = "Tanimoto"
params.max_stereocenters = 4
params.ewindow = 10
params.maxconfs = 200
params.rocs_input = ""
"""

from __future__ import annotations

__all__ = ["ROCSSimilarity"]
import copy
from typing import List, Optional
import logging

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .rocs.rocs_similarity import ROCSOverlay
from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger(__name__)


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    rocs_input: List[str]
    color_weight: List[float]
    shape_weight: List[float]
    max_stereocenters: List[int]
    ewindow: List[int]
    maxconfs: List[int]
    similarity_measure: List[str]
    custom_cff: Optional[List[str]] = Field(default_factory=lambda: [None])


@add_tag("__component")
class ROCSSimilarity:
    def __init__(self, params: Parameters):
        self.rocs_input = params.rocs_input[0]
        self.color_weight = params.color_weight[0]
        self.shape_weight = params.shape_weight[0]
        self.max_stereocenters = params.max_stereocenters[0]
        self.ewindow = params.ewindow[0]
        self.maxconfs = params.maxconfs[0]
        self.similarity_measure = params.similarity_measure[0]
        self.custom_cff = params.custom_cff[0]
        self.rocs = rocs = ROCSOverlay(
            rocs_input=self.rocs_input,
            color_weight=self.color_weight,
            shape_weight=self.shape_weight,
            max_stereocenters=self.max_stereocenters,
            ewindow=self.ewindow,
            maxconfs=self.maxconfs,
            similarity_measure=self.similarity_measure,
            custom_cff=self.custom_cff,
        )

        self.number_of_endpoints = len(params.rocs_input)
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.ndarray:
        scores = []
        results = self.rocs.calculate_rocs_score(smilies)
        scores.append(results)
        return ComponentResults(scores)
