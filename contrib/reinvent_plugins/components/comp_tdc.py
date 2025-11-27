"""Compute scores with Therapeutics Data Commons

Example TOML config (staged learning):

[[stage.scoring.component]]

[stage.scoring.component.TDCommons]
[[stage.scoring.component.TDCommons.endpoint]]
name = "tdc_jnk3"
weight = 1

[stage.scoring.component.TDCommons.endpoint.params]
type = "JNK3"

"""

from __future__ import annotations

__all__ = ["TDCommons"]
from dataclasses import dataclass, field
from typing import List
import logging

from tdc import Oracle
import numpy as np
from rdkit import Chem

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output

from importlib.metadata import version

logger = logging.getLogger("reinvent")

# Practical Molecular Benchmark oracles
supported_types = ["ALBUTEROL_SIMILARITY",
                   "AMLODIPINE_MPO", 
                   "CELECOXIB_REDISCOVERY",
                   "DECO_HOP",
                   "DRD2", 
                   "FEXOFENADINE_MPO",
                   "GSK3B",
                   "ISOMERS_C7H8N2O2",
                   "ISOMERS_C9H10N2O2PF2CL",
                   "JNK3",
                   "MEDIAN1",
                   "MEDIAN2",
                   "MESTRANOL_SIMILARITY", 
                   "OSIMERTINIB_MPO",
                   "PERINDOPRIL_MPO",
                   "QED",
                   "RANOLAZINE_MPO",
                   "SCAFFOLD_HOP",
                   "SITAGLIPTIN_MPO",
                   "THIOTHIXENE_REDISCOVERY", 
                   "TROGLITAZONE_REDISCOVERY",
                   "VALSARTAN_SMARTS",
                   "ZALEPLON_MPO"]


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    type: List[str]


@add_tag("__component")
class TDCommons:
    def __init__(self, params: Parameters):
        logger.info(f"Using TD Commons version {version('pytdc')}")
        self.oracles = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        for type in params.type:

            type_uppercase = type.upper()
            if type_uppercase not in supported_types:
                raise NotImplementedError(
                    f"TDCommons type {type_uppercase} is not supported."
                )

            with suppress_output():
                oracle = Oracle(name=type_uppercase)

                self.oracles.append(oracle)

    # TODO: normalize smiles does not keep all by default
    def __call__(self, smilies: List[str]) -> np.array:

        cleaned_smilies = self.normalize(smilies)

        assert len(cleaned_smilies) == len(smilies)

        scores = []

        for oracle in self.oracles:

            with suppress_output():
                pred = oracle(cleaned_smilies)

            assert len(cleaned_smilies) == len(smilies)

            assert len(pred) == len(cleaned_smilies)

            scores.append(
                np.array(
                    [
                        p if smi is not "INVALID" else np.nan
                        for p, smi in zip(pred, cleaned_smilies)
                    ]
                )
            )

        return ComponentResults(scores)

    def normalize(self, smilies: List[str]) -> List:
        """Convert to RDKit smiles. Remove annotations from SMILES

        :param smilies: list of SMILES strings
        """

        cleaned_smilies = []

        for smiles in smilies:
            mol = Chem.MolFromSmiles(smiles)

            # If not None
            if not mol:
                cleaned_smilies.append("INVALID")

                logger.warning(f"{__name__}: {smiles} could not be converted")

                continue

            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)

            cleaned_smilies.append(Chem.MolToSmiles(mol))

        return cleaned_smilies
