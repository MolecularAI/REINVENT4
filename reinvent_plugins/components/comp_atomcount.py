"""Compute scores with Carbon atom Count:)"""

from __future__ import annotations
from rdkit import Chem

__all__ = ["AtomCount"]
from typing import List
import logging
import numpy as np
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    target: List[str]


@add_tag("__component")
class AtomCount:
    """
    Calculates the number of target atoms in a SMILES and gives reward accordingly.
    Gives reward of 0 if given SMILES is invalid molecule.
    DISCLAIMER: Counts both aromatic and aliphatic atoms.
    """

    def __init__(self, params: Parameters):
        self.targets = params.target
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies):

        carbon_counts = []

        for smi in smilies:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                carbon_counts.append(np.nan)
                continue

            count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in self.targets)
            carbon_counts.append(float(count))

        scores = [np.array(carbon_counts, dtype=float)]

        return ComponentResults(scores=scores)