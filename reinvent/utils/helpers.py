from __future__ import annotations

__all__ = [
    "get_cuda_driver_version",
    "set_seed",
    "extract_sections",
    "write_json_config",
    "get_tokens_from_vocabulary",
]
import os
import random
import subprocess as sp
from typing import Optional
import logging

import numpy as np
import torch

from reinvent.utils import config_parse

logger = logging.getLogger(__name__)


def get_cuda_driver_version() -> Optional[str]:
    """Get the CUDA driver version via modinfo if possible.

    This is for Linux only.

    :returns: driver version or None
    """

    # Alternative
    # result = sp.run(["/usr/bin/nvidia-smi"], shell=False, capture_output=True)
    # if "Driver Version:" in str_line:
    #    version = str_line.split()[5]

    try:
        result = sp.run(["/sbin/modinfo", "nvidia"], shell=False, capture_output=True)
    except Exception:
        return

    for line in result.stdout.splitlines():
        str_line = line.decode()

        if str_line.startswith("version:"):
            cuda_driver_version = str_line.split()[1]

            return cuda_driver_version


def set_seed(seed: int):
    """Set global seed for reproducibility

    :param seed: the seed to initialize the random generators
    """

    if seed is None:
        return

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_sections(config: dict) -> dict:
    """Extract the sections of a config file

    :param config: the config file
    :returns: the extracted sections
    """

    # FIXME: stages are a list of dicts in RL, may clash with global lists
    return {k: v for k, v in config.items() if isinstance(v, (dict, list))}


def write_json_config(global_dict, json_out_config):
    def dummy(config):
        global_dict.update(config)
        config_parse.write_json(global_dict, json_out_config)

    return dummy


def get_tokens_from_vocabulary(vocabulary) -> tuple(set):
    """Get the tokens supported by a model's vocabulary.

    :param vocabulary: model's vocabylary object
    :returns: 2-tuple of tokens
    """

    logger.debug(f"{__name__}: {type(vocabulary)=}")

    if hasattr(vocabulary, "tokens"):  # Reinvent
        tokens1 = set(vocabulary.tokens())
        tokens2 = set()
    elif hasattr(vocabulary, "decoration_vocabulary"):  # Libinvent
        tokens1 = set(vocabulary.decoration_vocabulary.tokens())
        tokens2 = set(vocabulary.scaffold_vocabulary.tokens())
    elif hasattr(vocabulary, "input"):  # Linkinvent
        tokens1 = set(vocabulary.input.vocabulary.tokens())
        tokens2 = set(vocabulary.target.vocabulary.tokens())
    elif "tokens" in vocabulary:  # Transformer models
        tokens1 = set(vocabulary["tokens"])
        tokens2 = set()
    else:
        raise RuntimeError(f"Unknown vocabulary type: {type(vocabulary)=}")

    return (tokens1, tokens2)
