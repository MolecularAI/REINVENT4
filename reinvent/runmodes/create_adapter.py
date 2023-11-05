"""Create a model adapter from a Torch pickle file"""

__all__ = ["create_adapter"]
from typing import Tuple
import logging

import torch

from reinvent import models


logger = logging.getLogger(__name__)


def create_adapter(dict_filename: str, mode: str, device: torch.device) -> Tuple:
    """Read a dict from a Torch pickle find and return an adapter and model dict.

    :param dict_filename: filename of the Torch pickle file
    :param mode: "training" or "inference" mode
    :param device: torch device
    :returns: the adapter class, the model type
    """

    # FIXME: check if map_location is needed for CPU as in Reinvent
    save_dict = torch.load(dict_filename, map_location=device)

    if "model_type" in dir(save_dict):
        model_type = save_dict._model_type
    else:  # heuristics
        # FIXME: ugly if
        if "network" in save_dict:
            model_type = "Reinvent"
        elif "model" in save_dict:
            model_type = "Libinvent"
        elif "encoder_params" in save_dict["network_parameter"]:
            model_type = "Linkinvent"
        elif "num_heads" in save_dict["network_parameter"]:
            model_type = "Mol2Mol"
        else:
            model_type = None

    try:
        adapter_class = getattr(models, f"{model_type}Adapter")
        model_class = getattr(models, f"{model_type}Model")
    except AttributeError:
        msg = f"Unknown model type: {model_type}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    model = model_class.create_from_dict(save_dict, mode, device)
    adapter = adapter_class(model)

    return adapter, save_dict, model_type
