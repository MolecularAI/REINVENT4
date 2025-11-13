"""Scoring component for generic Maize workflows

Calls Maize workflows in a serialized format, utilizing the`ReinventEntry`
and `ReinventExit` nodes to provide a consistent JSON interface.

Example use:

.. code-block:: toml

  [stage.scoring.component.Maize]
  params.executable = "/home/me/miniforge3/envs/maize-dev/bin/maize"
  params.workflow = "workflow.yaml"
  params.config = "maize_global_config.toml"  # optional
  params.log = "maize-docking.log"  # optional

  [[stage.scoring.component.Maize.endpoint]]
  name = "Maize Docking"
  weight = 1.0

  params.property = "r_i_docking_score"  # IMPORTANT: set parameters in YAML file

  transform.low = -13.5
  transform.high = -6
  transform.k = 0.2
  transform.type = "reverse_sigmoid"

  # Overwrite parameters in YAML file
  [stage.scoring.component.Maize.endpoint.params.parameters]
  save_poses = false
  host = "localhost"
  n_jobs = 7

Using the following Maize workflow definition
(in YAML format, but TOML and JSON also work):

.. code-block:: yaml

   # REINVENT-Maize interface example
   #
   # Usage:
   # maize ./workflow.yml --inp input.json --out out.json
   #
   # Input:
   #
   # {
   #     "smiles": ["CCO", "CCF"],
   #     "metadata": {"iteration": 0}
   # }
   #
   # Output:
   # {
   #     "scores": [1.0, 2.0]
   # }

   name: Docking
   level: INFO

   nodes:

   # Entry point node accepting SMILES (and optional metadata) in a JSON file
   - name: smiles
     type: ReinventEntry

   # Main subgraph, accepts list[str] and outputs list[IsomerCollection]
   - name: dock
     type: Docking

   # Exit point, extracts scores from list[IsomerCollection], creates JSON output
   - name: rnv
     type: ReinventExit

   # ReinventEntry can output optional metadata, if not needed, sent to a Void node
   - name: void
     type: Void

   # Normal linear workflow topology
   channels:
   - sending:
       smiles: out
     receiving:
       dock: inp
   - sending:
       dock: out
     receiving:
       rnv: inp
   - sending:
       smiles: out_metadata
     receiving:
       void: inp

   parameters:

   # This is the required input parameter
   - name: inp
     map:
     - smiles: data

   # Receptor to use for docking
   - name: receptor
     value: "./receptor.pdbqt"
     map:
     - dock: receptor

   # Pocket location
   - name: center
     value: [3.3, 11.5, 24.8]
     map:
     - dock: search_center

   # JSON output for REINVENT
   - name: out
     map:

   # Ask ReinventExit to return all scores
   # Use property in REINVENT TOML file to select the wanted one
   - name: scores
    value: __return_all
    map:
      - rnv: tag
"""

from __future__ import annotations

__all__ = ["Maize"]

import os
import shlex
import tempfile
from typing import Any
import logging

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.run_program import run_command
from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles
from .utils import (
    create_extra_parameters,
    prepare_input_json,
    wait_for_output,
    parse_output,
    COMPATIBILITY_KEY,
)


logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    :param executable: Path to Maize executable
    :param workflow: Path to Maize workflow definition
    :param debug: Whether to print additional debug information (optional)
    :param keep: Whether to keep all temporary workflow files (optional)
    :param log: Path to Maize logfile (optional)
    :param config: Path to Maize system configuration (optional)
    :param parameters: Dictionary containing workflow parameters to override (optional)
    :param skip_normalize: skip normalize_smiles (optional)
    :param skip_on_failure: skip component on failure (optional)
    :param pass_fragments: take fragmented smiles as input instead of complete smiles (optional)
    :param property: Name of the property to return (optional)
    """

    executable: list[str]
    workflow: list[str]
    debug: list[bool] = Field(default_factory=lambda: [False])
    keep: list[bool] = Field(default_factory=lambda: [False])
    log: list[str | None] = Field(default_factory=lambda: [None])
    config: list[str | None] = Field(default_factory=lambda: [None])
    parameters: list[dict[str, Any]] = Field(default_factory=lambda: [{}])
    skip_normalize: list[bool] = Field(default_factory=lambda: [False])
    skip_on_failure: list[bool] = Field(default_factory=lambda: [True])
    pass_fragments: list[bool] = Field(default_factory=lambda: [False])

    property: list[str] = Field(default_factory=lambda: [COMPATIBILITY_KEY])


CMD = "{exe} {config} --inp {inp} --out {out} --parameters {params}"


@add_tag("__component")
class Maize:
    """Scoring component for the Maize workflow manager"""

    def __init__(self, params: Parameters):
        # We only allow one call to maize and one, single workflow per component
        self.executable = os.path.abspath(params.executable[0])
        self.workflow = os.path.abspath(params.workflow[0])
        self.debug = params.debug[0]
        self.keep = params.keep[0]
        self.log = os.path.abspath(params.log[0])
        self.config = os.path.abspath(params.config[0])
        self.skip_on_failure = params.skip_on_failure[0]
        self.skip_normalize = params.skip_normalize[0]
        self.pass_fragments = params.pass_fragments[0]
        self.parameters = params.parameters[0]

        # Each endpoint can have a property of its own
        self.properties = params.property

        self.step_id = 1
        self.number_of_endpoints = len(self.properties)
        self.smiles_type = "rdkit_smiles"

        logger.info(f"Using maize executable {self.executable} with workflow "
                    f"{self.workflow}")
        logger.info(f"Requested maize score names: "
                    f"{', '.join([p for p in self.properties])}")

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> np.array:
        with tempfile.TemporaryDirectory() as tmp:
            in_filename = os.path.join(tmp, "maize_input.json")
            out_filename = os.path.join(tmp, "maize_output.json")
            extra_filename = os.path.join(tmp, "maize_extra.json")

            prepare_input_json(in_filename, smilies, self.step_id)
            create_extra_parameters(extra_filename, self.parameters)

            self.step_id += 1

            command = shlex.split(
                CMD.format(
                    exe=self.executable,
                    config=self.workflow,
                    inp=in_filename,
                    out=out_filename,
                    params=extra_filename,
                )
            )

            if self.debug:
                command.append("--debug")

            if self.keep:
                command.append("--keep")

            if self.log:
                command.extend(["--log", self.log])

            if self.config:
                command.extend(["--config", self.config])

            all_scores = []

            try:
                run_command(command)
            except ValueError as err:
                if self.skip_on_failure:
                    logger.warning(
                        f"{__name__}: failed, returning zero for all compounds, original error: {err}"
                    )

                    # FIXME: zeroes for all compounds
                    for _ in self.properties:
                        all_scores.append(np.full(len(smilies), 0.0, dtype=np.float32))

                    return ComponentResults(scores=all_scores)
                raise

            data = wait_for_output(out_filename)

            for key in self.properties:
                compat_mode = False

                if self.number_of_endpoints == 1 and key == COMPATIBILITY_KEY:
                    compat_mode = True
                else:
                    logger.info(f"Known maize scores: {', '.join([key for key in data['scores']])}")

                endpoint_scores = parse_output(data, key, compat_mode)

                # FIXME: The NaN to zero conversion here is to deal with docking
                #        in case many/most/all docking fails.  Needs
                #        reconsideration as to how to handle this properly
                scores = np.nan_to_num(np.array(endpoint_scores))
                all_scores.append(scores)

        return ComponentResults(scores=all_scores)
