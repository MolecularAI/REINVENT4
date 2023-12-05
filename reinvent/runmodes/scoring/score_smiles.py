"""Score SMILES with a selected scoring function"""

from __future__ import annotations

__all__ = ["score_smiles_from_file"]

import numpy as np

from reinvent.scoring.scorer import Scorer
from reinvent.config_parse import read_smiles_csv_file
from reinvent.chemistry import Conversions
from reinvent.chemistry.enums import FilterTypesEnum
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer


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

    # DEFAULT = "default"  # all the below
    # NEUTRALIZE_CHARGES = "neutralise_charges"
    # GET_LARGEST_FRAGMENT = "get_largest_fragment"
    # REMOVE_HYDROGENS = "remove_hydrogens"
    # REMOVE_SALTS = "remove_salts"
    # GENERAL_CLEANUP = "general_cleanup"
    # TOKEN_FILTERS = "token_filters"
    # VOCABULARY_FILTER = "vocabulary_filter"
    # VALID_SIZE = "valid_size"
    # HEAVY_ATOM_FILTER = "heavy_atom_filter"
    # ALLOWED_ATOM_TYPE = "allowed_atom_type"
    # ALIPHATIC_CHAIN_FILTER = "aliphatic_chain_filter"
    filter_types = FilterTypesEnum()
    standardizer_config = [  # this will need fine-tuning depending on downstream component
        FilterConfiguration(filter_types.GET_LARGEST_FRAGMENT),
        FilterConfiguration(filter_types.GENERAL_CLEANUP)
    ]
    standardizer = RDKitStandardizer(standardizer_config, isomeric=True)

    if standardize:
        actions.append(standardizer.apply_filter)

    # NOTE: setting to True may result in problems down-streams
    if randomize:
        actions.append(conversions.randomize_smiles)

    smilies = read_smiles_csv_file(csv_filename, columns=0, actions=actions)

    mask = np.full(len(smilies), True, dtype=bool)
    scorer = Scorer(config)
    results = scorer(smilies, mask, mask)

    return results
