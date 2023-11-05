"""Score SMILES with a selected scoring function"""

from __future__ import annotations

__all__ = ["score_smiles_from_file"]
from typing import List, TYPE_CHECKING

import numpy as np

from reinvent.scoring.scorer import Scorer
from reinvent.config_parse import read_smiles_csv_file
from reinvent.chemistry import Conversions
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer


if TYPE_CHECKING:
    from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration


def score_smiles_from_file(
    csv_filename: str,
    config: dict,
    standardize: bool = True,
    randomize: bool = False,
):
    """Score SMILES from a file with a given scoring function

    :param csv_filename: CSV filename containing SMILES
    :param config: the scoring configuration
    :param standardize: whether to standardize the SMILES
    :param randomize: whether to randomize the SMILES
    """

    actions = []
    conversions = Conversions()
    standardizer_config: List[FilterConfiguration] = []
    standardizer = RDKitStandardizer(standardizer_config)

    if standardize:
        actions.append(standardizer.apply_filter)

    # NOTE: setting to True may result in problems down-streams
    if randomize:
        actions.append(conversions.randomize_smiles)

    smilies = read_smiles_csv_file(csv_filename, columns=0, actions=actions)

    mask = np.full(len(smilies), True, dtype=bool)
    scorer = Scorer(config)
    results = scorer(smilies, mask)

    return results
