"""259 descriptors from LillyMol's iwdescr

NOTE: iwdescr will terminate on the first invalid SMILES
"""

__all__ = ["LillyDescriptors"]

import csv
import shlex
from io import StringIO
import operator
from dataclasses import dataclass
from typing import List
import logging

import numpy as np

from reinvent_plugins.normalize import normalize_smiles
from ..run_program import run_command
from ..component_results import ComponentResults
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

# -O controls which descriptor subset is computed but only one allowed
DESCR_CMD = "{topdir}/bin/Linux/iwdescr -E autocreate -A D -g all -O all -i smi -"  # assume in path


@add_tag("__parameters")
@dataclass
class Parameters:
    descriptors: List[List[str]]
    topddir: List[str]


@add_tag("__component", "filter")
class LillyDescriptors:
    def __init__(self, params: Parameters):
        self.descriptors = params.descriptors[0]  # does not make sense to have multiple endpoints

        descr_cmd = DESCR_CMD.format(topdir=params.topddir[0])
        self.descr_cmd = shlex.split(descr_cmd)

        self.smiles_type = "lilly_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        result = run_command(self.descr_cmd, input="\n".join(smilies))
        scores = parse_output(result.stdout, self.descriptors, len(smilies))

        return ComponentResults(scores)


def parse_output(lines: str, cols: List[str], nsmilies: int) -> List[np.ndarray[float]]:
    """Parse the output from iwdescr and extract the desired columns."""

    file = StringIO(lines)

    header = file.readline().strip().split(" ")
    idx = [header.index(col) for col in cols if col in header]

    reader = csv.reader(file, delimiter=" ")
    rows = {}

    for row in reader:
        ID = int(row[0].replace("IWD", "")) - 1
        rows[ID] = [float(item) for item in operator.itemgetter(*idx)(row)]

    for i in range(nsmilies):
        if i not in rows.keys():
            rows[i] = np.full(len(idx), np.nan)

    if len(rows) != nsmilies:
        logger.warning(f"{__name__}: Processed only {len(row)} of {nsmiles} SMILES")

    scores = []

    for col in zip(*rows.values()):
        scores.append(np.array(col))

    return scores
