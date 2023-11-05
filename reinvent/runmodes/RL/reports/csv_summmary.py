"""Write out a CSV summary"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


csv_logger = logging.getLogger("csv")


@dataclass
class CSVSummary:
    step: int
    score_results: ScoreResults
    prior_nll: float
    agent_nll: float
    augmented_nll: float
    scaffolds: list


def write_summary(data: CSVSummary, write_header=False) -> tuple:
    """Write a summary to the CSV logger

    :param data: data to be written
    :param write_header: whether to write the header
    :returns: headers and columns
    """

    header = ["Agent", "Prior", "Target", "Score", "SMILES"]
    results = data.score_results

    columns = [
        data.agent_nll,
        data.prior_nll,
        data.augmented_nll,
        results.total_scores,
        results.smilies,
    ]

    if data.scaffolds:
        header.append("Scaffold")
        columns.append(data.scaffolds)

    names = []
    scores = []
    raw_scores = []

    for transformed_result in results.completed_components:
        names.extend(transformed_result.component_names)

        for transformed_scores in transformed_result.transformed_scores:
            scores.append(transformed_scores)

        for original_scores in transformed_result.component_result.scores:
            raw_scores.append(original_scores)    

    for name, score, raw_score in zip(names, scores, raw_scores):
        header.extend((name, f"{name} (raw)"))
        columns.extend((score, raw_score))

    if write_header:
        csv_logger.info(header + ["step"])

    for row in zip(*columns):
        csv_logger.info(row + (data.step,))

    return header, columns
