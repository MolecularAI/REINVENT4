"""Synthetic accessibility score

Synthetic Accessibility Score of Drug-like Molecules from P. Ertl & A. Schuffenhauer, J Cheminformatics 1:8 (2009)
Larger SA score = more difficult to synthesize
"""

from __future__ import annotations

__all__ = ["SAScore"]

from typing import List

import numpy as np

from .sascorer import calculateScore
from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag


@add_tag("__component")
class SAScore:
    def __init__(self, *args, **kwargs):
        # no user.configurable parameters
        pass

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.ndarray:
        sa_scores = [calculateScore(mol) for mol in mols]

        return ComponentResults([np.array(sa_scores)])
