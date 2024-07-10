"""Topological polar surface area (TPSA) descriptor from RDKit

This implementation allows to include polar S and P as described in
https://www.rdkit.org/docs/RDKit_Book.html#implementation-of-the-tpsa-descriptor
"""

from __future__ import annotations

__all__ = ["TPSA"]

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component"""

    includeSandP: Optional[List[bool]] = Field(default_factory=lambda: [False])


@add_tag("__component")
class TPSA:
    def __init__(self, params: Parameters):
        self.includeSandP = params.includeSandP
        self.number_of_endpoints = len(params.includeSandP)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.ndarray:
        scores = []

        for includeSandP in self.includeSandP:
            endpoint_scores = []

            for mol in mols:
                try:
                    score = Descriptors.TPSA(mol, includeSandP=includeSandP)
                except ValueError:
                    score = np.nan

                endpoint_scores.append(score)

            scores.append(np.array(endpoint_scores, dtype=float))

        return ComponentResults(scores)
