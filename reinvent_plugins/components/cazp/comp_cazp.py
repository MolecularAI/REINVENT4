"""Compute easy-of-synthesis scores with AiZynthFinder.

For AiZynthFinder documentation, see:
- https://github.com/MolecularAI/aizynthfinder

This implementations calls external command.
The external command, in turn, can call AiZynthFinder command-line interface,
or run CLI in parallel on a cluster, or even call REST API for AiZynth.
External commands are left out of this repository.

Simplest external command, with AiZynthFinder installed,
would be the following bash script::

  #!/usr/bin/env bash

  cp /dev/stdin > input.smi && \
  aizynthcli \
     --smiles input.smi \
     --config config.yml \
     --output output.json.gz \
     >>aizynth-stdout-stderr.txt 2>&1 && \
  zcat output.json.gz

"""

__all__ = ["Cazp"]

import logging

from reinvent_plugins.normalize import normalize_smiles
from reinvent_plugins.components.cazp.parameters import Parameters, split_params

from reinvent_plugins.components.cazp.runner import run_aizynth

from ..add_tag import add_tag
from ..component_results import ComponentResults

logger = logging.getLogger("reinvent")


@add_tag("__component")
class Cazp:
    def __init__(self, params: Parameters):
        self.params = params
        self.smiles_type = "rdkit_smiles"  # For the normalizer.
        logger.info("Initializing CAZP")
        self.params_run, self.endpoints = split_params(params)
        self.steps = 0
        self.number_of_endpoints = len(self.endpoints)

        logger.info(f"CAZP params: {params}, run params: {self.params_run}")

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> ComponentResults:
        """Returns AiZynth score.

        This function assumes it will start one CAZP run,
        but we can extract multiple endpoints from one and the same run.
        """

        self.steps += 1
        out = run_aizynth(smilies, self.params_run, self.steps)

        # Here we can iterate over endpoints.
        # Component results are returned as a list of numpy arrays.
        # Each numpy array should be the size of `smilies`,
        # i.e. it should contain all scores for one endpoint.
        # Multiple endpoints go into the list as separate numpy arrays.
        all_scores = []
        for endpoint in self.endpoints:
            scores = endpoint.get_scores(smilies, out)
            all_scores.append(scores)

        return ComponentResults(
            scores=all_scores,
        )
