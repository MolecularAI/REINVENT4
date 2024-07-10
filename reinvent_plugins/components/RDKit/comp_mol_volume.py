"""Compute the RDKit molecule volume from a conformer and its grid-encoding

The quality depends on the quality of the conformer.
"""

from typing import List, Optional

import numpy as np
from rdkit.Chem import AllChem as Chem
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent_plugins.mol_cache import molcache


@add_tag("__parameters")
@dataclass
class Parameters:
    grid_spacing: Optional[List[float]] = Field(default_factory=lambda: [0.2])
    box_margin: Optional[List[float]] = Field(default_factory=lambda: [2.0])


@add_tag("__component")
class MolVolume:
    """Compute the molecular volume from the grid of the molecule's shape"""

    def __init__(self, params: Parameters):
        self.grid_spacings = params.grid_spacing
        self.box_margin = params.box_margin
        self.number_of_endpoints = len(params.grid_spacing)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for grid_spacing, box_margin in zip(self.grid_spacings, self.box_margin):
            volumes = []

            for mol in mols:
                try:
                    mol3d = Chem.AddHs(mol)  # will not consider protonation state
                    Chem.EmbedMolecule(mol3d)
                    volume = Chem.ComputeMolVolume(
                        mol3d, gridSpacing=grid_spacing, boxMargin=box_margin
                    )
                except ValueError:
                    volume = np.nan

                volumes.append(volume)

            scores.append(np.array(volumes, dtype=float))

        return ComponentResults(scores)
