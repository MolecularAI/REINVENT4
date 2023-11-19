"""NIBR substructure filter

Simple implementation using a cutoff to delineate between "good" and "bad"
molecules.  Could be implemented as a normal scoring component using
a transform.

This is a demonstration on how one could build a filter catalog with the help of RDKit.
The actual code was taken from RDKit commit 4a69bc3493dd3e9bb9f7a519e306fbcb545f1452
and adapted as needed.  The original CSV file was converted to pickle file.
"""

__all__ = ["NIBRSubstructureFilters"]
import os
import pickle
from dataclasses import dataclass
from typing import List
import logging

import numpy as np
from rdkit import Chem

from .assign_filters import assign_filters
from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    cutoff: List[int]


@add_tag("__component", "filter")
class NIBRSubstructureFilters:
    def __init__(self, params: Parameters):
        path = os.path.dirname(__file__)
        catalog_filename = os.path.join(path, "catalog.pkl")

        with open(catalog_filename, "rb") as pfile:
            self.catalog = pickle.load(pfile)

        self.cutoffs = params.cutoff

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for cutoff in self.cutoffs:
            nibr_scores = assign_filters(self.catalog, mols)

            # SubstructureMatches, Min_N_O_filter, Frac_N_O, Covalent,
            # SpecialMol, SeverityScore
            scores.append(
                np.array([entry.SeverityScore < cutoff for entry in nibr_scores], dtype=int)
            )

        return ComponentResults(scores)
