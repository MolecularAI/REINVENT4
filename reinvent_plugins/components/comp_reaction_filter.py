"""Reaction filter

This is primarily for Lib/Linkinvent and was previously directly in their
scoring methods in RL.
"""

__all__ = ["ReactionFilter"]
from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem

from reinvent.chemistry.library_design import reaction_filters
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    type: List[str]  # filter type
    reaction_smarts: List[List[str]]  # RDKit reaction SMARTS


@dataclass
class ReactionFilterParams:
    reactions: List[str]


CONV_MAP = {
    "selective": "Selective",
    "nonselective": "NonSelective",
    "definedselective": "DefinedSelective",
}


@add_tag("__component", "filter")
class ReactionFilter:
    def __init__(self, params: Parameters):
        self.reaction_filters = []

        for reaction_type, reaction_smarts in zip(params.type, params.reaction_smarts):
            temp = reaction_type.lower()
            name = f"{CONV_MAP[temp]}Filter"

            filter_class = getattr(reaction_filters, name)
            params = ReactionFilterParams(reaction_smarts)

            self.reaction_filters.append(filter_class(params))

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for reaction_filter in self.reaction_filters:
            reaction_scores = [reaction_filter.evaluate(mol) if mol else np.nan for mol in mols]
            scores.append(reaction_scores)

        return ComponentResults([np.array(arr, dtype=float) for arr in scores])
