"""Run an external scoring subprocess

Run external process: provide specific command line parameters when needed
pass on the SMILES as a series of strings at the end.
"""

from __future__ import annotations

__all__ = ["ExternalProcess"]

import os
import shlex
import json
from typing import List
import logging
import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag

SKIP_EXEC = "/dev/null"

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

    executable: List[str]
    args: List[str]
    property: List[str]


@add_tag("__component")
class ExternalProcess:
    """Calculate the scores by running an external process

    The (optional) arguments to the executable are followed by the SMILES
    string by string.  The executable is expected to return the scores
    in a JSON list.  E.g. run a conda script (predict.py) in a specific
    environment (qptuna):

    specific_parameters.executable = "/home/user/miniconda3/condabin/conda"
    specific_parameters.args = "run -n qptuna python predict.py"
    specific_parameters.property = "predictions"

    And predict.py as

    import sys
    import pickle
    import json

    smilies = sys.stdin.readlines()

    with open('model.pkl', 'rb') as pf:
        model = pickle.load(pf)

    scores = model.predict_from_smiles(smilies)
    data = {"version": 1, "payload": {"predictions": list(scores)}}

    print(json.dumps(list(scores)))
    """

    def __init__(self, params: Parameters):

        self.executable = params.executable[0]
        self.args = params.args[0]

        # Ensure only one executable is used for all endpoints
        for exe, arg in zip(params.executable, params.args):
            if (exe, arg) != (self.executable, self.args):
                raise ValueError(
                    f"{__name__}: Only one executable and arguments per ExternalProcess scoring component is supported. "
                    f"Got '{self.executable} {self.args}' and '{exe} {arg}'. "
                    f"For multiple executables, separate them into multiple components."
                )

        self.properties = params.property
        self.number_of_endpoints = len(self.properties)

    def __call__(self, smilies: List[str]) -> ComponentResults:

        _executable = os.path.abspath(self.executable)
        _args = shlex.split(self.args)
        smiles_input = "\n".join(smilies)

        result = run_command([_executable] + _args, input=smiles_input)
        data = json.loads(result.stdout)

        if "payload" not in data:
            raise ValueError(
                f"{__name__}: Stdout from {self.executable} does not contain 'payload': {result.stdout}"
            )

        payload = data["payload"]

        scores = []
        for property in self.properties:

            if property not in payload:
                raise ValueError(
                    f"{__name__}: Payload from {self.executable} does not contain '{property}': {payload}"
                )

            # Extract property as scores.
            scores.append(np.array(payload[property]))

        # Extract all other keys as metadata (all keys in payload except the properties).
        metadata = {k: v for k, v in payload.items() if k not in self.properties}
        logger.debug(
            f"{__name__}: Executed '{_executable} {self.args}' with {len(smilies)} SMILES, "
            f"extracted properties: {self.properties}, metadata: {metadata}"
        )

        return ComponentResults(scores, metadata=metadata)
