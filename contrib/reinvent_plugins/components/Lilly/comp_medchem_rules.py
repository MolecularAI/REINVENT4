"""Iain Watson's Lilly Medchem rules

Compute the demerit score for each SMILES.  Not implemented as a filter.
"""

__all__ = ["LillyMedchemRules"]
import re
import tempfile
from dataclasses import dataclass
from typing import List
import logging

import numpy as np
from rdkit import Chem

from ..run_program import run_command
from ..component_results import ComponentResults
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

CMD = "Lilly_Medchem_Rules.rb"  # assume in path


@add_tag("__parameters")
@dataclass
class Parameters:
    relaxed: List[bool]  # 7-50 heavy atoms, 160 demerit cutoff


@add_tag("__component")
class LillyMedchemRules:
    def __init__(self, params: Parameters):
        self.want_relaxed = params.relaxed

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        for want_relaxed in self.want_relaxed:
            cmd = [CMD]

            if want_relaxed:
                cmd.append("-relaxed")

            with (tempfile.NamedTemporaryFile(mode="w+", suffix=".smi", delete=True) as in_smi,):
                for num, smiles in enumerate(smilies):
                    in_smi.write(f"{smiles} ID{num}\n")

                in_smi.flush()

                cmd.append(in_smi.name)
                result = run_command(cmd)

            demerits = parse_output(result.stdout, smilies)
            scores.append(np.array(demerits, dtype=float))

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


def parse_output(lines: str, smilies: List[str]) -> List[float]:
    """Parse the output from the medchem rule tool

    Note: RDKit canonicalization of the Lilly SMILES may not string match
          with the original canonical SMILES because Lilly has its own
          chemistry model!  Hence match by ID.
    """

    # bad0.smi: mc_first_pass
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
    with open("bad3.smi", "r") as mc_bad:
        for line in mc_bad:
            match = DEMERIT_PATTERN.match(line)
            ID = match.group(2)
            demerit = match.group(4)

            # FIXME: this needs reconsideration because the demerits may
            #        actually be very low but the compounds was rejected
            #        maybe increase demerit? user adjustable?
            mc_smilies[ID] = int(demerit)

    for line in lines.splitlines():
        match = DEMERIT_PATTERN.match(line)
        ID = match.group(2)
        demerit = match.group(4)

        if not match.group(3):  # not demerits
            mc_smilies[ID] = 0
        else:
            mc_smilies[ID] = int(demerit)

    scores = list(dict(sorted(mc_smilies.items())).values())

    return scores
