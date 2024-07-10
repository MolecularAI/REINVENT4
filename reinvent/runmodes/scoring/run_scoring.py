"""Sore a list of SMILES

Passes a list of SMILES to a scoring function of choice.
"""

from __future__ import annotations

__all__ = ["run_scoring"]
import os
import re
from collections import Counter
import logging
from typing import List, Tuple, Callable

import numpy as np

from reinvent.scoring.scorer import Scorer
from reinvent.chemistry.standardization.filter_types_enum import FilterTypesEnum
from reinvent.runmodes.scoring.file_io import TabFileReader, write_csv
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer
from .validation import ScoringConfig

logger = logging.getLogger(__name__)

ACTION_CALLABLES = List[Callable[[str], str]]
REINVENT_SMILES_COLUMN = "RDKit_SMILES (REINVENT)"
SUFFIX_PATTERN = r"^(.*)\.\d+$"


def run_scoring(input_config: dict, write_config: str = None, *args, **kwargs) -> None:
    """Scoring run setup.

    :param input_config: configuration
    """

    config = ScoringConfig(**input_config)
    parameters = config.parameters

    output_csv_filename = os.path.abspath(parameters.output_csv)
    smiles_filename = os.path.abspath(parameters.smiles_file)
    smiles_column = parameters.smiles_column
    standardize = parameters.standardize_smiles

    if smiles_column == REINVENT_SMILES_COLUMN:
        raise RuntimeError(f"{__name__}: the column name {REINVENT_SMILES_COLUMN} is reserved")

    logger.info(f"Scoring SMILES from file {smiles_filename}")

    if standardize:
        logger.info("Standardizing input SMILES")

    actions = setup_actions(standardize=standardize)
    reader = read_data(smiles_filename, actions=actions, smiles_column=smiles_column)

    mask = np.array([True if smiles else False for smiles in reader.smilies])

    scoring_function = Scorer(config.scoring)

    if callable(write_config):
        write_config(config.model_dump())

    results = scoring_function(reader.smilies, mask, mask)

    logger.info(f"Number of SMILES processed: {len(results.smilies)}")

    results_header, results_rows = get_result_table(results)
    header, rows = merge_columns(reader, results_header, results_rows)

    logger.info(f"Writing scoring results to {output_csv_filename}")

    write_csv(output_csv_filename, header, rows)


def merge_columns(
    reader: TabFileReader, results_header: List, results_rows: List
) -> Tuple[List[str], List[List]]:
    """Merge the columns from the input file with the columns from the scoring results

    :param reader: the reader objets with the table from the input file
    :param results_header: the header from the scoring results
    :param results_rows: all rows from the scoring results:
    :returns: the constructed table
    """

    # retain original SMILES name and replace if needed
    header = uniquify_header(reader.header_line + results_header)

    rows = []

    for orig_row, results_row in zip(reader.rows, results_rows):
        rows.append(orig_row + results_row)

    return header, rows


def get_result_table(results) -> Tuple[List, List[List]]:
    """Extract the columns of the scoring results

    :param results: the scoring results
    :returns: the header line and the rows
    """
    total_scores = results.total_scores
    names = []
    scores = []
    raw_scores = []

    for transformed_result in results.completed_components:
        names.extend(transformed_result.component_names)

        for transformed_scores in transformed_result.transformed_scores:
            scores.append(transformed_scores)

        for original_scores in transformed_result.component_result.scores:
            raw_scores.append(original_scores)

    scores = zip(*scores)
    raw_scores = zip(*raw_scores)

    header = [REINVENT_SMILES_COLUMN, "Score"] + names + [f"{name} (raw)" for name in names]

    rows = []

    for smiles, total_score, _scores, _raw_scores in zip(
        results.smilies, total_scores, scores, raw_scores
    ):
        row = [smiles, total_score]
        row.extend(_scores)
        row.extend(_raw_scores)

        rows.append(row)

    return header, rows


def setup_actions(standardize: bool = True) -> ACTION_CALLABLES:
    """Setup for actions on SMILES

    FIXME: randomization should not be needed here

    :param standardize: whether to standardize the SMILES
    :returns: list of actions
    """

    actions = []

    # List of known standardization flags
    # DEFAULT = "default"  # all of the below
    # NEUTRALIZE_CHARGES, GET_LARGEST_FRAGMENT, REMOVE_HYDROGENS, REMOVE_SALTS,
    # GENERAL_CLEANUP, UNWANTED_PATTERNS, VOCABULARY_FILTER, VALID_SIZE,
    # HEAVY_ATOM_FILTER, ALLOWED_ELEMENTS
    filter_types = FilterTypesEnum()
    standardizer_config = [  # this will need fine-tuning depending on downstream component
        FilterConfiguration(filter_types.GET_LARGEST_FRAGMENT),
        FilterConfiguration(filter_types.GENERAL_CLEANUP),
    ]
    standardizer = RDKitStandardizer(standardizer_config, isomeric=True)

    if standardize:
        actions.append(standardizer.apply_filter)

    return actions


def read_data(filename: str, actions: ACTION_CALLABLES, smiles_column: str) -> TabFileReader:
    """Score SMILES from a file with a given scoring function

    :param filename: CSV filename containing SMILES
    :param actions: list of actions for each SMILES
    :param smiles_column: name of the SMILES column
    :returns: a reader object containing the read data
    """

    if filename.endswith(".csv"):
        header = True
    elif filename.endswith(".smi"):
        header = False
    else:
        raise RuntimeError(f"Unknown file format: {filename}")

    reader = TabFileReader(filename, header=header, actions=actions, smiles_column=smiles_column)
    reader.read()

    return reader


def uniquify_header(header: List[str]) -> List[str]:
    """Uniquify the strings in a list

    :param header: header with potentially multiple occurrences of a string
    """

    # normalize strings, assuming the suffix pattern is never used by the user
    clean_header = [re.sub(SUFFIX_PATTERN, r"\1", name) for name in header]

    counts = Counter(clean_header)

    new_header = []
    counters = {k: 1 for k in counts.keys()}

    for name in clean_header:
        count = counts[name]

        if count > 1:
            num = counters[name]
            new_name = f"{name}.{num}" if num > 1 else name
            counters[name] += 1
        else:
            new_name = name

        new_header.append(new_name)

    return new_header
