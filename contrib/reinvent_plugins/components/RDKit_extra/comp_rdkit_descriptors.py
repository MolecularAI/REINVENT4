"""Compute a desired list of RDKit descriptors up to a total of 210"""

__all__ = ["RDKitDescriptors"]
from dataclasses import dataclass
from typing import List
import logging

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
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
        self.calc = MolecularDescriptorCalculator(params.descriptor).CalcDescriptors

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []

        for mol in mols:
            result = self.calc(mol, missingVal=np.NaN)
            scores.append(np.array(result))

        scores = np.array(scores).transpose()

        return ComponentResults(list(scores))
