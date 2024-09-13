"""Run an external scoring subprocess

Run external process: provide specific command line parameters when needed
pass on the SMILES as a series of strings at the end.
"""

from __future__ import annotations

__all__ = ["DockStream"]

import os
import logging
import copy

from typing import List, Optional

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger(__name__)


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    configuration_path: List[str]  # should we type hint paths? List[Path]
    docker_script_path: List[str]
    docker_python_path: List[str]


@add_tag("__component")
class DockStream:
    """Run docking with DockStream (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00563-7)

    All specific configuration settings (grid, ligand preparation etc) are in the configuration file.
    Consistent with previous behaviour, cases where no docking pose is produced
    are given a score of zero
    """

    def __init__(self, params: Parameters):
        self._internal_step = 0
        self.docker_python_path = params.docker_python_path[0]
        self.docker_script_path = params.docker_script_path[0]
        self.configuration_path = params.configuration_path[0]
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        scores = []
        command = [
            self.docker_python_path,
            self.docker_script_path,
            "-conf",
            self.configuration_path,
            "-output_prefix",
            str(self._internal_step),
            "-smiles",
            ";".join(smilies),
            "-print_scores",
        ]

        result = run_command(command)
        docker_scores = result.stdout.split()
        # note: some ligands might have failed in along the way (embedding or docking) although they are valid molecules
        #       -> "docker.py" will return "NA"'s for failed molecules, as '0' could be a perfectly normal value;
        #       anything that cannot be cast to a floating point number will result in '0'

        for score in docker_scores:
            try:
                score = float(score)
            except ValueError:
                score = 0.0
            scores.append(score)

        self._internal_step += 1

        return ComponentResults([scores])
