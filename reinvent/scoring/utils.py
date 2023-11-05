"""Auxiliary functions."""

__all__ = ["suppress_output"]
from os import devnull
import re
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def suppress_output():
    """Context manager to redirect stdout and stderr to /dev/null"""

    with open(devnull, "w") as nowhere:
        with redirect_stderr(nowhere) as err, redirect_stdout(nowhere) as out:
            yield (err, out)


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
