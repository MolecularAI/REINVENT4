"""Tanimoto similarity and Jaccard distance"""

from __future__ import annotations

__all__ = ["TanimotoSimilarity", "TanimotoDistance"]

import warnings
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from reinvent.chemistry import conversions
from reinvent.chemistry.similarity import calculate_tanimoto_batch
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
class TanimotoSimilarity:
    """Compute the Tanimoto similarity

    Scoring component to compute the Tanimoto similarity between the provided
    SMILES and the generated molecule.  Supports fingerprint radius, count
    fingerprints and the use of pharmacophore-like features (see
    https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z).
    """

    def __init__(self, params: Parameters):
        self.fp_params = []

        for smilies, radius, use_counts, use_features in zip(
            params.smiles, params.radius, params.use_counts, params.use_features
        ):
            fingerprints = conversions.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            if not fingerprints:
                raise ValueError(f"{__name__}: unable to convert any SMILES to fingerprints")

            self.fp_params.append((fingerprints, radius, use_counts, use_features))

        self.number_of_endpoints = len(params.smiles)

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for fingerprints, radius, use_counts, use_features in self.fp_params:
            query_fingerprints = conversions.smiles_to_fingerprints(
                smilies, radius=radius, use_counts=use_counts, use_features=use_features
            )

            scores.extend(
                [
                    calculate_tanimoto_batch(fingerprint, query_fingerprints)
                    for fingerprint in fingerprints
                ]
            )

        return ComponentResults(scores)


# for backward compatibility
@add_tag("__component")
class TanimotoDistance(TanimotoSimilarity):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TanimotoDistance is deprecated and will be removed in a future release. "
            "Please use TanimotoSimilarity instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
