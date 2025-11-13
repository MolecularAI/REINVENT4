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

__all__ = ["SynthSense", "CAZP"]

import logging
import dataclasses

from reinvent_plugins.normalize import normalize_smiles
from reinvent_plugins.components.synthsense.parameters import Parameters, split_params

from reinvent_plugins.components.synthsense.runner import run_aizynth

from ..add_tag import add_tag
from ..component_results import ComponentResults

logger = logging.getLogger("reinvent")


@add_tag("__component")
class SynthSense:
    def __init__(self, params: Parameters):
        self.params = params
        self.smiles_type = "rdkit_smiles"  # For the normalizer.
        logger.info("Initializing synthsense")
            
        self.params_run, self.endpoints = split_params(params)
        self.steps = 0
        self.number_of_endpoints = len(self.endpoints)
        
        # Set no_cache based on whether any endpoint requires it
        # Only endpoints with batch-dependent scoring need cache disabled
        self.no_cache = any(endpoint.no_cache for endpoint in self.endpoints)
        logger.info(f"synthsense cache disabled: {self.no_cache} (based on endpoint requirements)")

        logger.info(f"synthsense params: {params}, run params: {self.params_run}")

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> ComponentResults:
        """Returns AiZynthFinder score.

        This function assumes only one AiZynthFinder job is run
        but we can extract multiple endpoints from one and the same run.
        """

        self.steps += 1
        out = run_aizynth(smilies, self.params_run, self.steps)

        all_scores = []

        for endpoint in self.endpoints:
            scores = endpoint.get_scores(smilies, out)
            all_scores.append(scores)

        return ComponentResults(scores=all_scores)


# for backward compatibility
@add_tag("__component")
class CAZP(SynthSense):
    pass
