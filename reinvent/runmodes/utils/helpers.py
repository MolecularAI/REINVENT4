"""Some auxiliary functionality that does not fit anywhere else.

FIXME: may need to move some/all of these
"""

from __future__ import annotations

__all__ = ["disable_gradients", "set_torch_device"]
import logging
from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter


logger = logging.getLogger(__name__)


def disable_gradients(model: ModelAdapter) -> None:
    """Disable gradient tracking for all parameters in a model

    :param model: the model for which all gradient tracking will be switched off
    """

    for param in model.get_network_parameters():
        param.requires_grad = False


def set_torch_device(args_device: str = None, device: str = None) -> torch.device:
    """Set the Torch device

    :param args_device: device name from the command line
    :param device: device name from the config
    """

    logger.debug(f"{device=} {args_device=}")

    # NOTE: ChemProp > 1.5 would need "spawn" but hits performance 4-5 times
    #       Windows requires "spawn"
    # torch.multiprocessing.set_start_method('fork')

    if args_device:  # command line overwrites config file
        # NOTE: this will throw a RuntimeError if the device is not available
        torch.set_default_device(args_device)
        actual_device = torch.device(args_device)
    elif device:
        torch.set_default_device(device)
        actual_device = torch.device(device)
    else:  # we assume there are no other devices...
        torch.set_default_device("cpu")
        actual_device = torch.device("cpu")

    logger.debug(f"{actual_device=}")

    return actual_device
