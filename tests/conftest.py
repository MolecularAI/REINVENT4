"""Fixture setup for pytest

IMPORTANT: NEVER import directly from conftest.py
           also DO NOT mess with sys.path
"""

import os
import sys
import json
import pytest

from reinvent import models

# kludge to fix tests by-passing compatibility mode in the code
sys.modules["reinvent.models.mol2mol.models.vocabulary"] = models.transformer.core.vocabulary


def pytest_addoption(parser):
    """Command line parsing for pytest

    Use e.g. like
    $ pytest [pytest parameters and options] --device cpu
    """

    parser.addoption(
        "--device", default="cuda", choices=["cuda", "cpu"], help="set the torch device"
    )

    parser.addoption(
        "--json",
        default=os.environ.get("REINVENT_TEST_JSON"),  # FIXME: find a better mechanism
        help="JSON test " "configuration file",
    )


@pytest.fixture
def device(request):
    """Allow access to args.device in a unittest class

    Accessible in the unittest class as self.device
    Requires the class to be decorated with @pytest.mark.usefixtures("device")
    """

    # FIXME: is there a better way?
    try:
        request.cls.device = request.config.getoption("--device")
    except AttributeError:  # assume we are run from a function
        request.device = request.config.getoption("--device")
        return request.device


@pytest.fixture
def json_config(request):

    json_config = request.config.getoption("--json")

    if json_config is None:
        raise ValueError(
            "Missing json config."
            " Either pass it with --json option when calling pytest,"
            " or set REINVENT_TEST_JSON environmental variable."
        )

    with open(json_config) as jfile:
        json_input = jfile.read().replace("\r", "").replace("\n", "")
        config = json.loads(json_input)

    try:
        request.cls.json_config = config
    except AttributeError:  # assume we are run from a function
        request.json_config = config

    for key, value in config.get("ENVIRONMENTAL_VARIABLES", {}).items():
        if key not in os.environ:
            os.environ[key] = value

    return config
