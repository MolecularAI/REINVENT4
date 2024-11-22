"""
Scoring component for generic Maize workflows

This component will call Maize workflows in a serialized format, utilizing the
`ReinventEntry` and `ReinventExit` nodes to provide a consistent JSON interface.
This requires at least Maize 0.3.3 and Maize-contrib 0.2.1. Here's an example use:

.. code-block:: toml

   [stage.scoring.component.Maize]
   [[stage.scoring.component.Maize.endpoint]]
   name = "maize"
   weight = 10

   # Lower is better because we're docking
   transform.type = "reverse_sigmoid"
   transform.high = -3
   transform.low = -9.5
   transform.k = 0.5

   # Path to the maize executable
   params.executable = "/path/to/maize"

   # Path to the maize workflow
   params.workflow = "./workflow.yml"

   # Debug mode
   params.debug = true

   # Keep graph data
   params.keep = true

   # Config to use
   params.config = "/path/to/.config/maize.toml"

   # Additional workflow parameters, these will override
   # values specified in the workflow definition
   params.parameters.receptor = "/path/to/receptor.pdbqt"

   # Logs
   params.log = "/path/to/maize.log"

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

   # This is an entrypoint node accepting SMILES
   # (and optional metadata) in a JSON file
   - name: smiles
     type: ReinventEntry

   # Main calculation subgraph accepting list[str]
   # and outputting list[IsomerCollection]
   - name: dock
     type: Docking

   # Exit point, extracts scores from list[IsomerCollection]
   # and creates an output JSON
   - name: rnv
     type: ReinventExit

   # ReinventEntry can output optional metadata, if it's
   # not required in can be sent to a Void node
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
     - rnv: data

"""

from __future__ import annotations

__all__ = ["Maize"]

import os
import shlex
import json
import tempfile
import time
from typing import List, Any

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.

    :param executable: Path to Maize executable
    :param workflow: Path to Maize workflow definition
    :param debug: Whether to print additional debug information (optional)
    :param keep: Whether to keep all temporary workflow files (optional)
    :param log: Path to Maize logfile (optional)
    :param config: Path to Maize system configuration (optional)
    :param parameters: Dictionary containing workflow parameters to override (optional)

    """

    executable: List[str]
    workflow: List[str]
    debug: List[bool] = Field(default_factory=lambda: [False])
    keep: List[bool] = Field(default_factory=lambda: [False])
    log: List[str | None] = Field(default_factory=lambda: [None])
    config: List[str | None] = Field(default_factory=lambda: [None])
    parameters: List[dict[str, Any]] = Field(default_factory=lambda: [{}])


CMD = "{exe} {config} --inp {inp} --out {out} --parameters {params}"


@add_tag("__component")
class Maize:
    """Scoring component for the Maize workflow manager"""

    def __init__(self, params: Parameters):
        self.executable = params.executable[0]
        self.workflow = params.workflow[0]
        self.debug = params.debug[0]
        self.keep = params.keep[0]
        self.log = params.log[0]
        self.config = params.config[0]
        self.parameters = params.parameters[0]
        self.smiles_type = "rdkit_smiles"

        if len(params.workflow) > 1:
            raise ValueError("The Maize component currently only supports a single endpoint")
        self.step_id = 0

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        with tempfile.TemporaryDirectory() as tmp:
            in_json = os.path.join(tmp, "input.json")
            out_json = os.path.join(tmp, "output.json")
            extra_params = os.path.join(tmp, "extra.json")

            prepare_input_json(in_json, smilies, self.step_id)
            create_extra_parameters(extra_params, self.parameters)

            self.step_id += 1
            command = shlex.split(
                CMD.format(
                    exe=self.executable,
                    config=self.workflow,
                    inp=in_json,
                    out=out_json,
                    params=extra_params,
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

            _ = run_command(command)
            wait_for_output(out_json, sleep_for=3)

            endpoint_score = parse_output(out_json)

            scores = [np.nan_to_num(np.array(endpoint_score))]

        return ComponentResults(scores=scores)


def create_extra_parameters(file: str, parameters: dict[str, Any]) -> None:
    """Creates a JSON file with additional maize workflow parameters"""
    with open(file, "w", encoding="utf-8") as flow:
        json.dump(parameters, flow, indent=4)


def prepare_input_json(json_file: str, smilies: List[str], step_id: int) -> None:
    """Prepare JSON ReinventEntry node input"""
    ids = [str(idx) for idx in range(len(smilies))]

    # Metadata entry reserved for later
    data = {"names": ids, "smiles": smilies, "metadata": {"iteration": step_id}}
    with open(json_file, "w", encoding="utf-8") as inp:
        inp.write(json.dumps(data))


def wait_for_output(filename: str, sleep_for: float) -> None:
    """Wait for output to be generated by Maize"""
    while not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        time.sleep(sleep_for)


def parse_output(filename: str) -> List[float]:
    """Output looks like:
    {
        "scores": [...],
    }
    """

    if not os.path.isfile(filename):
        raise ValueError(f"{__name__}: failed, missing output file")

    with open(filename, "r", encoding="utf-8") as jfile:
        data = json.load(jfile)

    if "scores" not in data:
        raise ValueError(f"{__name__}: JSON file does not contain 'scores'")

    return data["scores"]
