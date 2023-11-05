"""Sore a list of SMILES

Passes a list of SMILES to a scoring function of choice.
"""

__all__ = ["run_scoring"]
import os
import logging

from reinvent import setup_logger, CsvFormatter
from reinvent.runmodes.scoring.score_smiles import score_smiles_from_file


logger = logging.getLogger(__name__)


def run_scoring(config: dict, *args, **kwargs) -> None:
    """Scoring run setup.

    :param config: configuration
    """

    smiles_filename = os.path.abspath(config["parameters"]["smiles_file"])
    output_csv_filename = os.path.abspath(config.get("output_csv", "_scoring.csv"))
    sf_section = config["scoring"]

    csv_logger = setup_logger(
        name="csv",
        filename=output_csv_filename,
        formatter=CsvFormatter(),
        propagate=False,
        level="INFO",
    )

    logger.info(f"Scoring SMILES from file {smiles_filename}")

    results = score_smiles_from_file(smiles_filename, sf_section, standardize=True, randomize=False)

    smilies = results.smilies

    logger.info(f"Number of SMILES processed: {len(smilies)}")

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

    logger.info(f"Writing scoring results to {output_csv_filename}")

    header = ["SMILES", "Score"] + names + [f"{name} (raw)" for name in names]
    csv_logger.info(header)

    for smiles, total_score, _scores, _raw_scores in zip(smilies, total_scores, scores, raw_scores):
        row = [smiles, total_score]
        row.extend(_scores)
        row.extend(_raw_scores)

        csv_logger.info(row)
