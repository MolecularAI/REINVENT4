"""Group count

Allow groups of atoms defined by SMARTS only a certain number of times in a
molecules.
"""

from __future__ import annotations

__all__ = ["MatchingSubstructure"]

from typing import List

from rdkit import Chem
import numpy as np
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
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

    smarts: List[str | List[str]]
    use_chirality: List[bool]


@add_tag("__component", "penalty")
class MatchingSubstructure:
    def __init__(self, params: Parameters):
        self.patterns_per_endpoint = []
        self.use_chirality = params.use_chirality

        for smarts in params.smarts:
            patterns = []

            if isinstance(smarts, list):
                for smart in smarts:
                    patterns.append(Chem.MolFromSmarts(smart))
            else:
                patterns = [Chem.MolFromSmarts(smarts)]

            if patterns:
                self.patterns_per_endpoint.append(patterns)

        if not self.patterns_per_endpoint:
            raise ValueError(f"{__name__}: no valid SMARTS patterns found")

        self.number_of_endpoints = len(params.smarts)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for patterns, use_chirality in zip(self.patterns_per_endpoint, self.use_chirality):
            match = []
            for mol in mols:
                if not mol:
                    match.append(np.nan)
                else:
                    match.append(
                        any(
                            mol.HasSubstructMatch(pattern, useChirality=use_chirality)
                            for pattern in patterns
                        )
                    )

            scores.append(0.5 * (1.0 + np.array(match, dtype=float)))

        return ComponentResults(scores)
