"""Lilly PAINS pattern to score for various assays

This uses the substructure tool to match for known PAINS patterns.  Matches
are not counted.
"""

__all__ = ["LillyPAINS"]
import os
import csv
import re
import operator
import shlex
from dataclasses import dataclass
from typing import List, Dict
import logging

import numpy as np

from reinvent_plugins.normalize import normalize_smiles
from ..run_program import run_command
from ..component_results import ComponentResults
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

LILLY_HOME = "LILLY_MOL_ROOT"
ACCEPTED_STEM = "accepted"
TSUB_CMD = (
    "{topdir}/bin/Linux/tsubstructure -E autocreate -b -u -i smi -o smi -A D -m - -m QDT "
    "-n {accepted} -q F:{topdir}/data/queries/PAINS/queries_latest -"
)
SCORES_FILENAME = os.path.join(os.path.dirname(__file__), "pains_scores.csv")
KNOWN_ASSAYS = [
    "Alpha",
    "ELISA",
    "FB",
    "FP",
    "FRET",
    "SPA",
    "OverallActivityEnrichment",
    "QCEnrichment",
    "Alpha_HS",
    "ELISA_HS",
    "FB_HS",
    "FP_HS",
    "FRET_HS",
    "SPA_HS",
    "HSEnrichment",
    "TotalScore",
]


@add_tag("__parameters")
@dataclass
class Parameters:
    assay: List[str]


@add_tag("__component")
class LillyPAINS:
    def __init__(self, params: Parameters):
        if any([assay not in KNOWN_ASSAYS for assay in params.assay]):
            raise RuntimeError(f"{__name__}: one or more assays not in {', '.join(KNOWN_ASSAYS)}")

        self.assays = params.assay

        if LILLY_HOME not in os.environ:
            raise RuntimeError(f"{__name__}: {LILLY_HOME} not in environment")

        self.pains_scores = read_scores_from_csv(SCORES_FILENAME)
        tsub_cmd = TSUB_CMD.format(topdir=os.environ[LILLY_HOME], accepted=ACCEPTED_STEM)
        self.tsub_cmd = shlex.split(tsub_cmd)

        self.smiles_type = "lilly_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        headers = self.pains_scores["_headers"]
        idx = [headers.index(assay_name) - 1 for assay_name in headers if assay_name in self.assays]

        smilies_ids = [f"{smiles} ID{num}\n" for num, smiles in enumerate(smilies)]
        result = run_command(self.tsub_cmd, input="\n".join(smilies_ids))
        scores = parse_output(result.stdout, idx, self.pains_scores, len(smilies))

        return ComponentResults(scores)


def read_scores_from_csv(filename) -> Dict[str, List[int]]:
    """Read the Lilly PAINS scores from a CSV file"""

    scores = {}

    with open(filename, "r") as cfile:
        reader = csv.reader(cfile)
        scores["_headers"] = next(reader, None)

        for row in reader:
            name = row[0]
            assay_scores = row[1:]
            scores[name] = [int(score) for score in assay_scores]

    return scores


# \1 SMILES ID
# \2 number of SMARTS pattern matches
# \3 name of pattern matched
REJECTED_PATTERN = re.compile(r".*? ID(\d+) \((\d+) matches to '(.*)'\)")
ACCEPTED_PATTERN = re.compile(r".* ID(\d+)")


def parse_output(
    lines: str, idx: List[int], pains_scores: Dict[str, List[int]], nsmilies: int
) -> List[np.ndarray[float]]:
    """Parse the output from tsubstructure and extract the desired columns."""

    if len(idx) > 1:
        get_rows = lambda row: [float(item) for item in operator.itemgetter(*idx)(row)]
    else:
        get_rows = lambda row: [float(name_scores[idx[0]])]

    rows = {}  # collect PAINS hits

    # FIXME: handle multiple matches
    for line in lines.splitlines():
        match = re.match(REJECTED_PATTERN, line)
        ID, _, name = match.groups()
        ID = int(ID)

        # Lookup groups
        name_scores = pains_scores[name]
        rows[ID] = get_rows(name_scores)

    accepted_ids = []

    with open(f"{ACCEPTED_STEM}.smi", "r") as accepted:
        for line in accepted:
            match = re.match(ACCEPTED_PATTERN, line)
            ID = int(match.group(1))
            accepted_ids.append(ID)

    for i in range(nsmilies):
        if i not in rows.keys():
            rows[i] = np.full(len(idx), np.nan)

        if i in accepted_ids:
            rows[i] = np.full(len(idx), 0.0)

    if len(rows) != nsmilies:
        logger.warning(f"{__name__}: Processed only {len(rows)} of {nsmilies} SMILES")

    sorted_rows = {k: rows[k] for k in sorted(rows)}

    scores = []

    for col in zip(*sorted_rows.values()):
        scores.append(np.array(col))

    return scores
