"""Create a model adapter from a Torch pickle file"""

__all__ = ["create_adapter"]
import pprint
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

    save_dict = torch.load(dict_filename, map_location="cpu")

    if "metadata" in save_dict:
        metadata: models.ModelMetaData = save_dict["metadata"]

        if not metadata.hash_id:
            logger.warning(f"{dict_filename} does not contain a hash ID")
        else:
            valid = models.check_valid_hash(save_dict)
            pp = pprint.PrettyPrinter(indent=2)

            if valid:
                logger.info(f"{dict_filename} has valid hash:\n{pp.pformat(metadata.as_dict())}")
            else:
                logger.error(f"{dict_filename} has invalid hash:\n{pp.pformat(metadata.as_dict())}")
    else:
        logger.warning(f"{dict_filename} does not contain metadata")

    if "model_type" in save_dict:
        model_type = save_dict["model_type"]
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

    compatibility(model)

    network_params = model.network.parameters()
    num_params = sum([tensor.numel() for tensor in network_params])
    logger.info(f"Number of network parameters: {num_params:,}")

    return adapter, save_dict, model_type

def compatibility(model):
    """Compatibility mode for old Mol2Mol priors"""

    from reinvent.models.mol2mol.models.vocabulary import Vocabulary

    if isinstance(model.vocabulary, Vocabulary):
        from reinvent.models.transformer.core import vocabulary as tvoc

        tokens = model.vocabulary._tokens
        model.vocabulary = tvoc.Vocabulary()
        model.vocabulary._tokens = tokens
