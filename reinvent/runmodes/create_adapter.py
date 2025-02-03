"""Create a model adapter from a Torch pickle file"""

__all__ = ["create_adapter"]

import os
import time
import uuid
import pathlib
import pprint
import logging

import torch

from reinvent import models
from reinvent.utils import prior_registry
from reinvent.models import meta_data

logger = logging.getLogger(__name__)


def create_adapter(dict_filename: str, mode: str, device: torch.device) -> tuple:
    """Read a dict from a Torch pickle find and return an adapter and model dict.

    :param dict_filename: filename of the Torch pickle file
    :param mode: "training" or "inference" mode
    :param device: torch device
    :returns: the adapter class, the model type
    """

    dict_filename = resolve_model_filename(dict_filename)
    save_dict = torch.load(dict_filename, map_location="cpu", weights_only=False)
    check_metadata(dict_filename, save_dict)

    if "model_type" in save_dict:
        model_type = save_dict["model_type"]

        # kludge to handle new-style transformers
        if model_type in ["Linkinvent", "Libinvent"] and save_dict.get("version", -1) == 2:
            model_type += "Transformer"
    else:
        model_type = orig_style_priors(save_dict)

    adapter_class = getattr(models, f"{model_type}Adapter", None)
    model_class = getattr(models, f"{model_type}Model", None)

    if not adapter_class or not model_class:
        msg = f"Unknown model type: {model_type}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    model = model_class.create_from_dict(save_dict, mode, device)
    adapter = adapter_class(model)

    compatibility_setup(model)

    network_params = model.network.parameters()
    num_params = sum([tensor.numel() for tensor in network_params])
    logger.info(f"Number of network parameters: {num_params:,}")
    logger.info(f"Network architecture:\n{model.network}")

    return adapter, save_dict, model_type


def resolve_model_filename(name: str) -> str:
    """Resolve the filename of the model from the name.

    The name may either be a key from a registry or directly a filename.

    :param name: the name of the model
    :returns: the filename of the model
    """

    filename = prior_registry.get(name, None)

    if not filename:
        filename = pathlib.Path(name).resolve()

    if not filename.exists() or not os.access(filename, os.R_OK):
        msg = f"model file {filename} is not accessible"
        logger.critical(msg)
        raise RuntimeError(msg)

    return str(filename)  # torch.load does not like pathlib.Path


def check_metadata(dict_filename: str, save_dict: dict) -> None:
    """Check the metadata of the save dict from a model file.

    CUrrently, only logs warnings or errors but does not terminate the run.

    :param dict_filename: model pickle file with the save dict
    :param save_dict: the save dict
    """

    if "metadata" in save_dict:
        if save_dict["metadata"] is not None:
            if isinstance(save_dict["metadata"], dict):
                metadata = models.ModelMetaData(**save_dict["metadata"])
            else:  # backward compatibility
                metadata = save_dict["metadata"]

            if not metadata.hash_id:
                logger.warning(f"{dict_filename} does not contain a hash ID")
            else:
                valid = models.check_valid_hash(save_dict)
                pp = pprint.PrettyPrinter(indent=2)
                pp_dict = pp.pformat(metadata.as_dict())

                if valid:
                    logger.info(f"{dict_filename} has valid hash:\n{pp_dict}")
                else:
                    logger.error(f"{dict_filename} has invalid hash:\n{pp_dict}")
        else:
            logger.warning(f"{dict_filename} contains empty metadata")

            save_dict["metadata"] = meta_data.ModelMetaData(
                hash_id=None,
                hash_id_format="",
                model_id=uuid.uuid4().hex,
                origina_data_source="unknown",
                creation_date=time.time(),
                comments=["REINVENT4"],
            )
    else:
        logger.warning(f"{dict_filename} does not contain metadata")

        save_dict["metadata"] = meta_data.ModelMetaData(
            hash_id=None,
            hash_id_format="",
            model_id=uuid.uuid4().hex,
            origina_data_source="unknown",
            creation_date=time.time(),
            comments=["REINVENT4"],
        )


def orig_style_priors(save_dict: dict) -> str:
    """Determine model type heuristically

    Originally, prior files did not contain any metadata so the model type
    must be guessed from the layout of the save dict.

    :param save_dict: the save dict
    :returns: the model type descriptor as a string
    """

    if "network" in save_dict:
        model_type = "Reinvent"
    elif "model" in save_dict:
        model_type = "Libinvent"
    elif "encoder_params" in save_dict["network_parameter"]:
        model_type = "Linkinvent"
    elif "num_heads" in save_dict["network_parameter"]:
        model_type = "Mol2Mol"
    else:
        model_type = ""

    return model_type


def compatibility_setup(model):
    """Compatibility mode for old Mol2Mol priors

    :param model: model adapter object
    """

    from reinvent.models.mol2mol.models.vocabulary import Vocabulary

    if isinstance(model.vocabulary, Vocabulary):
        from reinvent.models.transformer.core import vocabulary as tvoc

        tokens = model.vocabulary._tokens
        model.vocabulary = tvoc.Vocabulary()
        model.vocabulary._tokens = tokens
