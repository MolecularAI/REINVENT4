"""Iain Watson's Lilly Medchem rules

Compute the demerit score for each SMILES.  Could be implemented as a filter
using the default demerit of 100 or the relaxed demerit of 160 as boundary.
"""

__all__ = ["LillyMedchemRules"]
import os
import shlex
import re
import tempfile
from dataclasses import dataclass
from typing import List
import logging

import numpy as np

from reinvent_plugins.normalize import normalize_smiles
from ..run_program import run_command
from ..component_results import ComponentResults
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

LILLY_HOME = "LILLY_MEDCHEM_RULES_ROOT"
CMDS_FILENAME = os.path.join(os.path.dirname(__file__), "lcm_commands.lst")
PARAMS = ["-c smax=25 -c hmax=40", "-c smax=26 -c hmax=50 -f 160"]  # default, relaxed


@add_tag("__parameters")
@dataclass
class Parameters:
    relaxed: List[bool]  # could be used for filter


@add_tag("__component")
class LillyMedchemRules:
    def __init__(self, params: Parameters):
        self.relaxed = params.relaxed
        self.commands = []

        if LILLY_HOME not in os.environ:
            raise RuntimeError(f"{__name__}: {LILLY_HOME} not in environment")

        self.topdir = os.environ[LILLY_HOME]

        cmds = read_commands(CMDS_FILENAME)
        self.commands = [cmds] * len(self.relaxed)

        self.smiles_type = "lilly_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores = []

        for cmds, relaxed in zip(self.commands, self.relaxed):
            with (tempfile.NamedTemporaryFile(mode="w+", suffix=".smi", delete=True) as in_smi,):
                for num, smiles in enumerate(smilies):
                    in_smi.write(f"{smiles} ID{num}\n")

                in_smi.flush()

                result = run_pipeline(cmds, self.topdir, in_smi.name, PARAMS[relaxed])

            demerits = parse_output(result, smilies)

            scores.append(demerits)

        return ComponentResults(scores)


# \1 SMILES
# \2 ID number
# \3 reason for rejection
BAD_PATTERN = re.compile(r"(.*?) ID(\d+) (.*)")

# \1 SMILES
# \2 ID number
# \3 (demerit string, optional)
# \4 demerit value, optional
# \5 reason for demerits, optional
DEMERIT_PATTERN = re.compile(r"(.*?) ID(\d+)( : D\((\d+)\) (.*))?")
UNWANTED = 999


def parse_output(lines: str, smilies: List[str]) -> np.ndarray[float]:
    """Parse the output from the medchem rule tool

    Note: RDKit canonicalization of the Lilly SMILES may not string match
          with the original canonical SMILES because Lilly has its own
          chemistry model!  Hence match by ID.
    """

    # bad0.smi: mc_first_pass - perception, standardisation, validation, etc.
    mc_smilies = {}

    with open("bad0.smi", "r") as mc_bad:
        for line in mc_bad:
            match = BAD_PATTERN.match(line)
            ID = match.group(2)
            mc_smilies[ID] = UNWANTED

    # bad1.smi, bad2.smi: tsubstructure - rejections 1 and 2
    for filename in ("bad1.smi", "bad2.smi"):
        with open(filename, "r") as tsub_bad:
            for line in tsub_bad:
                match = BAD_PATTERN.match(line)
                ID = match.group(2)
                mc_smilies[ID] = UNWANTED

    # bad3.smi: iwdemerit
    #           SMILES >= threshold (-f flag) end up here
    with open("bad3.smi", "r") as mc_bad:
        for line in mc_bad:
            match = DEMERIT_PATTERN.match(line)
            ID = match.group(2)
            demerit = match.group(4)
            mc_smilies[ID] = int(demerit)

    good = []

    for line in lines.splitlines():
        match = DEMERIT_PATTERN.match(line)
        ID = match.group(2)
        demerit = match.group(4)

        if not match.group(3):  # no demerits
            mc_smilies[ID] = 0
        else:
            mc_smilies[ID] = int(demerit)

        good.append(demerit)

    scores = np.full(len(smilies), np.nan)

    for i, score in mc_smilies.items():
        idx = int(i)
        scores[idx] = score

    return scores


def read_commands(cmd_filename: str) -> List[str]:
    """Read the list of commands from file"""

    cmds = []

    with open(cmd_filename, "r") as cfile:
        for line in cfile:
            cmd = line.strip()

            if not cmd or cmd.startswith("#"):
                continue

            cmds.append(cmd)

    return cmds


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def run_pipeline(cmds: List, topdir: str, in_smi: str, demerit_params: str) -> str:
    """Run the pipeline of commands"""

    results = None

    for n, cmd in enumerate(cmds):
        if n == 0:
            instr = None
        else:
            instr = results.stdout

        cmds_string = cmd.format_map(
            SafeDict(topdir=topdir, in_smi=in_smi, demerit_params=demerit_params)
        )
        results = run_command(shlex.split(cmds_string), input=instr)

    return results.stdout
