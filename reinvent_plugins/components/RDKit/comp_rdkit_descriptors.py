"""Compute a desired list of RDKit descriptors up to a total of 210"""

__all__ = ["RDKitDescriptors"]
from typing import List
import logging

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import numpy as np
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

KNOWN_DESCRIPTORS = {d.lower(): d for d, _ in Descriptors._descList}


@add_tag("__parameters")
@dataclass
class Parameters:
    descriptor: List[str]


@add_tag("__component")
class RDKitDescriptors:
    def __init__(self, params: Parameters):
        # collect descriptor from all endpoints: only one descriptor per endpoint!
        descriptors = []

        for descriptor in params.descriptor:
            desc = descriptor.lower()

            if desc not in KNOWN_DESCRIPTORS:
                raise ValueError(f"{__name__}: unknown descriptor {desc}")

            descriptors.append(KNOWN_DESCRIPTORS[desc])

        self.calc = MolecularDescriptorCalculator(descriptors).CalcDescriptors

        self.number_of_endpoints = len(params.descriptor)

        logger.info(f"Known RDKit descriptors: {' '.join(KNOWN_DESCRIPTORS.keys())}")

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []

        for mol in mols:
            result = self.calc(mol, missingVal=np.NaN)
            scores.append(np.array(result))

        scores = np.array(scores).transpose()

        return ComponentResults(list(scores))
