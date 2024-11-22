"""Compute scores with ChemProp

[[component.ChemProp.endpoint]]
name = "ChemProp Score"
weight = 0.7

# component specific parameters
param.checkpoint_dir = "ChemProp/3CLPro_6w63"
param.rdkit_2d_normalized = true
param.target_column = "dG"

# transform
transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4

# In case of multiclass models add endpoints as needed and set the target_column
"""

from __future__ import annotations

__all__ = ["ChemProp"]
from typing import List
import logging

import chemprop
import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
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

    checkpoint_dir: List[str]
    rdkit_2d_normalized: List[bool]
    target_column: List[str]


@add_tag("__component")
class ChemProp:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        args = [
            "--checkpoint_dir",  # ChemProp models directory
            params.checkpoint_dir[0],
            "--test_path",  # required
            "/dev/null",
            "--preds_path",  # required
            "/dev/null",
        ]

        if params.rdkit_2d_normalized[0]:
            args.extend(["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"])

        with suppress_output():
            chemprop_args = chemprop.args.PredictArgs().parse_args(args)
            chemprop_model = chemprop.train.load_model(args=chemprop_args)

            self.chemprop_params = chemprop_model, chemprop_args

        target_columns = chemprop_model[-1]
        target_idx = []
        seen = set()

        for target_column in params.target_column:
            if target_column not in target_columns:
                msg = (
                    f"{__name__}: unknown target column {target_column} (known: "
                    f"{', '.join(target_columns)})"
                )
                logger.critical(msg)
                raise ValueError(msg)

            if target_column in seen:
                msg = f"{__name__}: target columns must be unique ({params.target_column})"
                logger.critical(msg)
                raise ValueError(msg)

            seen.add(target_column)

            target_idx.append(target_columns.index(target_column))

        self.keeps = np.array(target_idx)
        self.number_of_endpoints = len(self.keeps)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        smilies_list = [[smiles] for smiles in smilies]

        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=self.chemprop_params[0],
                smiles=smilies_list,
                args=self.chemprop_params[1],
                return_invalid_smiles=True,
                return_uncertainty=False,
            )

        scores = np.array(preds).transpose()[self.keeps]
        scores[scores == "Invalid SMILES"] = np.nan

        return ComponentResults(list(scores.astype(float)))
