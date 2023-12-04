"""Compute all the 210 RDKit descriptors and extract the desired one

NOTE: individual functions are available for these descriptors
"""

__all__ = ["RDKitDescriptors"]
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import logging

from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
import numpy as np

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    descriptor: List[str]


@add_tag("__component")
class RDKitDescriptors:
    def __init__(self, params: Parameters):
        # collect descriptor from all endpoints: only one descriptor per endpoint
        self.descriptors = params.descriptor

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []
        descriptor_scores = defaultdict(list)

        for descriptor in self.descriptors:
            for mol in mols:
                result = CalcMolDescriptors(mol, missingVal=np.NaN)
                descriptor_scores[descriptor].append(result[descriptor])

        for _scores in descriptor_scores.values():
            scores.append(np.array(_scores, dtype=float))

        return ComponentResults(scores)
