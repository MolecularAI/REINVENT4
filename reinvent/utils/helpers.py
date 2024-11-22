from __future__ import annotations

import os
import random

import subprocess as sp
from typing import Optional

import numpy as np
import torch

from reinvent.utils import config_parse


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
