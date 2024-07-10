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


def check_one_attachment_point_per_atom(mol):
    """
    Check if the molecule has no ambiguous reaction sites with multiple R-groups on one atom.
    This function checks for connected atoms with different molAtomMapNumber values in LibInvent molecules, for example
    [C:0]([N:1])[C:0], which indicates multiple attachment points on one atom and therefore
    reaction filters will not work/are ambiguous.
    Cases with separate attachment points, such as [C:0]([C:0])C[C:1][N:1], are allowed.
    :param mol: input molecule to check
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            bond_number = atom.GetProp("molAtomMapNumber")
            # now check all neighbors of this atom and see if they have a missmatched molAtomMapNumber value
            for neighbor in atom.GetNeighbors():
                if (
                    neighbor.HasProp("molAtomMapNumber")
                    and neighbor.GetProp("molAtomMapNumber") != bond_number
                ):
                    # if so, raise an error
                    raise ValueError(
                        "Reaction filter applied to input with multiple attachment points on one atom"
                    )


@add_tag("__component", "filter")
class ReactionFilter:
    def __init__(self, params: Parameters):
        self.reaction_filters = []

        for reaction_type, reaction_smarts in zip(params.type, params.reaction_smarts):
            temp = reaction_type.lower()
            name = f"{CONV_MAP[temp]}Filter"

            filter_class = getattr(reaction_filters, name)
            rf_params = ReactionFilterParams(reaction_smarts)

            self.reaction_filters.append(filter_class(rf_params))

        self.number_of_endpoints = len(params.reaction_smarts)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for reaction_filter in self.reaction_filters:
            reaction_scores = []
            for mol in mols:
                if mol:
                    check_one_attachment_point_per_atom(mol)
                    reaction_scores.append(reaction_filter.evaluate(mol))
                else:
                    reaction_scores.append(np.nan)

            scores.append(reaction_scores)

        return ComponentResults([np.array(arr, dtype=float) for arr in scores])
