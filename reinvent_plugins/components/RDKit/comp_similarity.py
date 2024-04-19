"""Tanimoto and Jaccard similarity"""

from __future__ import annotations

__all__ = ["TanimotoDistance"]

from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from reinvent.chemistry.conversions import Conversions
from reinvent.chemistry.similarity import Similarity
from ..component_results import ComponentResults
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    smiles: List[List[str]]
    radius: List[int]
    use_counts: List[bool]
    use_features: List[bool]


@add_tag("__component")
class TanimotoDistance:
    def __init__(self, params: Parameters):
        self.chem = Conversions()
        self.similarity = Similarity()
        self.fp_params = []

        for smilies, radius, use_counts, use_features in zip(
            params.smiles, params.radius, params.use_counts, params.use_features
        ):
            fingerprints = self.chem.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            if not fingerprints:
                raise RuntimeError(f"{__name__}: unable to convert any SMILES to fingerprints")

            self.fp_params.append((fingerprints, radius, use_counts, use_features))

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for fingerprints, radius, use_counts, use_features in self.fp_params:
            query_fingerprints = self.chem.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            scores.extend(
                [
                    self.similarity.calculate_tanimoto_batch(fingerprint, query_fingerprints)
                    for fingerprint in fingerprints
                ]
            )

        return ComponentResults(scores)
