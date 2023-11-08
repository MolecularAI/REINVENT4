"""NIBR substructure filter

This is a demonstration on how one could build a filter catalog with the help of RDKit.
The actual code was taken from RDKit commit 4a69bc3493dd3e9bb9f7a519e306fbcb545f1452
and adapted as needed.  The original CSV file was converted to pickle file.
"""

__all__ = ["NIBRSubstructureFilters"]
#from dataclasses import dataclass
import pickle
from typing import List
import logging

import numpy as np
from rdkit import Chem

from .assign_filters import assign_filters
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

logger = logging.getLogger('reinvent')


#@add_tag("__parameters")
#@dataclass
#class Parameters:
#    catalog: List[List[str]]


@add_tag("__component", "filter")
class NIBRSubstructureFilters:
    def __init__(self, params: Parameters):
        with open("catalog.pkl", "rb") as pfile:
            self.catalog = pickle.load(pfile)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        nibr_scores = assign_filters(self.catalog, mols)

        return ComponentResults([np.array(scores, dtype=float)])
