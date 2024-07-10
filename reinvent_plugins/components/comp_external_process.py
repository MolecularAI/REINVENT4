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

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag


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


@add_tag("__component")
class ExternalProcess:
    """Calculate the scores by running an external process

    The (optional) arguments to the executable are followed by the SMILES
    string by string.  The executable is expected to return the scores
    in a JSON list.  E.g. run a conda script (predict.py) in a specific
    environment (qptuna):

    specific_parameters.executable = "/home/user/miniconda3/condabin/conda"
    specific_parameters.args = "run -n qptuna python predict.py"

    And predict.py as

    import sys
    import pickle
    import json

    smilies = sys.stdin.readlines()

    with open('model.pkl', 'rb') as pf:
        model = pickle.load(pf)

    scores = model.predict_from_smiles(smilies)

    print(json.dumps(list(scores)))
    """

    def __init__(self, params: Parameters):
        # FIXME: multiple endpoints
        self.executables = params.executable
        self.args = params.args
        self.number_of_endpoints = len(params.executable)

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for executable, args in zip(self.executables, self.args):
            _executable = os.path.abspath(executable)
            _args = shlex.split(args)
            smiles_input = "\n".join(smilies)

            result = run_command([_executable] + _args, input=smiles_input)

            data = json.loads(result.stdout)

            # '{"version": 1, "payload": {"predictions": [1, 2, 3, 4, 5]}}'
            scores.append(np.array(data["payload"]["predictions"]))

        return ComponentResults(scores)
