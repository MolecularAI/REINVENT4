"""Compute the PMI score in RDKit"""

from typing import List

import numpy as np
from rdkit.Chem import AllChem as Chem
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent_plugins.mol_cache import molcache


@add_tag("__parameters")
@dataclass
class Parameters:
    property: List[str]


@add_tag("__component")
class PMI:
    """Compute the PMI index to characterize the dimensionality of a molecule

    See https://doi.org/10.4155%2Ffmc-2016-0095
    """

    def __init__(self, params: Parameters):
        self.properties = params.property

        if not self.properties:
            raise ValueError(f"{__name__}: need one or both of: 'npr1', 'npr2'")

        for prop in self.properties:
            if prop not in ("npr1", "npr2"):
                raise ValueError(f"{__name__}: unknown property '{prop}', need 'npr1' or 'npr2'")

        self.number_of_endpoints = len(params.property)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores1 = []
        scores2 = []

        for mol in mols:
            mol3d = Chem.AddHs(mol)
            embed_result = Chem.EmbedMolecule(mol3d)
            
            if embed_result == -1:  # embedding failed
                npr1 = np.nan
                npr2 = np.nan
            else:
                npr1 = Chem.CalcNPR1(mol3d)
                npr2 = Chem.CalcNPR2(mol3d)

            scores1.append(npr1)
            scores2.append(npr2)

        npr_scores = {
            "npr1": np.array(scores1, dtype=float),
            "npr2": np.array(scores2, dtype=float),
        }

        # one score array per endpoint, in the order the endpoints were configured
        scores = [npr_scores[prop] for prop in self.properties]

        return ComponentResults(scores)
