"""A decorator to ensure that torch parameters are on the correct device

This decorator makes sure that tensor and parameter arguments given to a
method are transferred to the right device given the device as it has been
set for the instance.
"""

__all__ = ["consistify"]
from functools import wraps

import torch


def set_device(arg, device):
    """Move arg to the right device if necessary

    :param device: device of the instance arg is an attribute of
    :param arg: torch tensor or parameter
    :return: arg moved to the right device if necessary
    """

    if isinstance(arg, (torch.Tensor, torch.nn.Parameter)):
        if arg.device != device:
            arg = arg.to(device)

    return arg


def consistify(method):
    """Decorator to move a method's arguments to the device of the instance"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        self = args[0]
        assert hasattr(self, "device")

        new_args, new_kwargs = [], {}

        for arg in args[1:]:
            new_args.append(set_device(arg, self.device))

        for name, arg in kwargs.items():
            new_kwargs[name] = set_device(arg, self.device)

        return method(self, *new_args, **new_kwargs)

    return wrapper
