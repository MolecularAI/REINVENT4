"""Compute all the 1613 2D Mordred and extract the desired one

NOTE: individual functions are available for these descriptors
NOTE2: Mordred needs to be updated to replace np.float with the modern equivalent
"""

__all__ = ["MordredDescriptors"]
from dataclasses import dataclass
from typing import List
import logging

from rdkit import Chem
from mordred import Calculator, descriptors

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    descriptor: List[str]
    nprocs: List[int]


@add_tag("__component")
class MordredDescriptors:
    def __init__(self, params: Parameters):
        # collect descriptor from all endpoints: only one descriptor per endpoint
        self.descriptors = params.descriptor

        self.nprocs = params.nprocs[0]
        self.calc = Calculator(descriptors, ignore_3D=True)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        df = self.calc.pandas(mols, nproc=self.nprocs, quiet=True)  # Pandas DataFrame
        df = df.astype(float)  # will turn Missing into NaN
        scores = df[self.descriptors].T.to_numpy()

        return ComponentResults(scores)
