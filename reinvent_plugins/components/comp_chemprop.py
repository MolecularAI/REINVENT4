"""Compute scores with ChemProp

scoring_function.type = "product"
scoring_function.parallel = false

[[stage.scoring_function.component]]

type = "chemprop"
name = "ChemProp Score"

weight = 0.7

# component specific parameters
param.checkpoint_dir = "ChemProp/3CLPro_6w63"
param.rdkit_2d_normalized = true

transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4
"""

from __future__ import annotations

__all__ = ["ChemProp"]
from typing import List
import logging

import chemprop
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles

logger = logging.getLogger('reinvent')


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
    rdkit_2d_normalized: List[bool] = Field(default_factory=lambda: [False])


@add_tag("__component")
class ChemProp:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")
        self.chemprop_params = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = 'rdkit_smiles'

        for checkpoint_dir, rdkit_2d_normalized in zip(
            params.checkpoint_dir, params.rdkit_2d_normalized
        ):
            args = [
                "--checkpoint_dir",  # ChemProp models directory
                checkpoint_dir,
                "--test_path",  # required
                "/dev/null",
                "--preds_path",  # required
                "/dev/null",
            ]

            if rdkit_2d_normalized:
                args.extend(
                    ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
                )

            with suppress_output():
                chemprop_args = chemprop.args.PredictArgs().parse_args(args)
                chemprop_model = chemprop.train.load_model(args=chemprop_args)

                self.chemprop_params.append((chemprop_model, chemprop_args))

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        smilies_list = [[smiles] for smiles in smilies]
        scores = []

        for model, args in self.chemprop_params:
            with suppress_output():
                preds = chemprop.train.make_predictions(
                    model_objects=model,
                    smiles=smilies_list,
                    args=args,
                    return_invalid_smiles=True,
                    return_uncertainty=False,
                )

            scores.append(
                np.array(
                    [val[0] if "Invalid SMILES" not in val else np.nan for val in preds],
                    dtype=float,
                )
            )

        return ComponentResults(scores)
