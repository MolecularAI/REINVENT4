"""Matched molecular pairs"""

from __future__ import annotations

__all__ = ["Icolos"]

import os
import shlex
import json
import tempfile
import time
from typing import List, IO

import numpy as np
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
    """

    name: List[str]
    executable: List[str]  # this probably doesn't changee
    config_file: List[int]


CMD = "{exe} -conf {config} --global_variables"
GLOBALS = "input_json_path:{inp} output_json_path:{out} step_id:{step}"


@add_tag("__component")
class Icolos:
    def __init__(self, params: Parameters):
        self.names = params.name
        self.executable = params.executable
        self.config_filenames = params.config_file
        self.step_id = 0
        self.number_of_endpoints = len(params.name)
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for name, exe, config in zip(self.names, self.executable, self.config_filenames):
            with (
                tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as in_json,
                tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as out_json,
            ):
                prepare_input_json(in_json, smilies)

                self.step_id += 1
                command = shlex.split(CMD.format(exe=exe, config=config))
                command.append(
                    GLOBALS.format(inp=in_json.name, out=out_json.name, step=self.step_id)
                )

                _ = run_command(command)
                wait_for_output(out_json.name, ntimes=5, sleep_for=3)

                endpoint_score = parse_output(out_json.name, name)
                scores.append(np.array(endpoint_score))

        return ComponentResults(scores)


def prepare_input_json(json_file: IO[str], smilies: List[str]) -> None:
    ids = [str(idx) for idx in range(len(smilies))]
    data = {"names": ids, "smiles": smilies}
    json.dump(data, json_file, indent=4)


def wait_for_output(filename, ntimes, sleep_for):
    for _ in range(ntimes):
        if os.path.isfile(filename) and os.path.getsize(filename) > 0:
            break
        else:
            time.sleep(sleep_for)


def parse_output(filename: str, name: str) -> List:
    """Output looks like:
    {
        "results": [{
            "values_key": "docking_score",
            "values": ["-5.88841", "-5.72676", "-7.30167"]},
                    {
            "values_key": "shape_similarity",
            "values": ["0.476677", "0.458017", "0.510676"]},
                    {
            "values_key": "esp_similarity",
            "values": ["0.107989", "0.119446", "0.100109"]}],
        "names": ["0", "1", "2"]
    }
    """

    if not os.path.isfile(filename):
        raise ValueError(f"{__name__}: failed, missing output file")

    with open(filename, "r") as jfile:
        data = json.load(jfile)

    # TODO: this should be properly validated
    if "results" not in data:
        raise ValueError(f"{__name__}: JSON file does not contain 'results'")

    # FIXME: check if scores are really in the same order as the SMILES
    for entry in data["results"]:
        if entry["values_key"] == name:
            return entry["values"]

    raise ValueError(f"{__name__}: JSON file does not contain scores for {name}")
