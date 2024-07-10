"""Compute scores with Qptuna models"""

from __future__ import annotations

__all__ = ["Qptuna"]
import pickle
from typing import List
import logging
import json

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

    model_file: List[str]


@add_tag("__component")
class Qptuna:
    def __init__(self, params: Parameters):
        self.models = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        for filename in params.model_file:
            model = load_model(filename)
            self.models.append(model)

        metadata = json.dumps(model.metadata, indent=2)
        logger.info(f"Qptuna model metadata:\n{metadata}")

        self.number_of_endpoints = len(params.model_file)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for model in self.models:
            # FIXME: change to True as soon as code is ready to deal with
            #        uncertainties
            model_scores = model.predict_from_smiles(smilies, uncert=False)
            scores.append(np.array(model_scores))

        return ComponentResults(scores)


def load_model(filename: str):
    """Load a Qptuna pickle model

    :param filename: pickle file name with Qptuna model
    :returns: a Qptuna model
    """

    with open(filename, "rb") as mfile:
        model = pickle.load(mfile)

    return model
