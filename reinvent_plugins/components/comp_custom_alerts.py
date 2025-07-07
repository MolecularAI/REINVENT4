"""Compute scores with RDKit's QED"""

__all__ = ["CustomAlerts"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass
import logging

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

logger = logging.getLogger(__name__)


@add_tag("__parameters")
@dataclass
class Parameters:
    smarts: List[List[str]]


@add_tag("__component", "filter")
class CustomAlerts:
    def __init__(self, params: Parameters):
        # FIXME: read from file?
        # to avoid multiple computations of the temple, do it at init only
        self.smarts = []
        self.templates = []
        for subst in params.smarts[0]:  # assume there is only one endpoint...
            template = Chem.MolFromSmarts(subst)
            if template is not None:
                self.templates.append(template)
                self.smarts.append(subst)
            else: # FIXME: logging from inside scoring plugins does not seem to show up?
                logger.warning(f"Skipping invalid template {subst}")

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:

        all_matches = []

        for mol in mols:
            matches = []
            if mol:
                # test for each substructure
                for subst, template in zip(self.smarts, self.templates):
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(subst)):
                        matches.append(subst)
            all_matches.append(matches)

        scores = [1 - any(m) for m in all_matches]

        return ComponentResults(
            [np.array(scores, dtype=float)], metadata={"matchting_patterns": all_matches}
        )
