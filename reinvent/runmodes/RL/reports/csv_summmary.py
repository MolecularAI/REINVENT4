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
    smiles_state: list


def write_summary(data: CSVSummary, write_header=False) -> tuple:
    """Write a summary to the CSV logger

    :param data: data to be written
    :param write_header: whether to write the header
    :returns: headers and columns
    """

    header = ["Agent", "Prior", "Target", "Score", "SMILES", "SMILES_state"]
    results = data.score_results

    columns = [
        [f"{score:.4f}" for score in data.agent_nll],
        [f"{score:.4f}" for score in data.prior_nll],
        [f"{score:.4f}" for score in data.augmented_nll],
        [f"{score:.7f}" for score in results.total_scores],
        results.smilies,
        [str(state.value) for state in data.smiles_state]
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
            _scores = []

            for score in transformed_scores:
                try:
                    _scores.append(f"{float(score):.7f}")
                except ValueError:
                    _scores.append(score)

            scores.append(_scores)

        for original_scores in transformed_result.component_result.scores:
            _scores = []

            for score in original_scores:
                try:
                    _scores.append(f"{float(score):.4f}")
                except ValueError:
                    _scores.append(score)

            raw_scores.append(_scores)

    for name, score, raw_score in zip(names, scores, raw_scores):
        header.extend((name, f"{name} (raw)"))
        columns.extend((score, raw_score))

    if write_header:
        csv_logger.info(header + ["step"])

    for row in zip(*columns):
        csv_logger.info(row + (data.step,))

    return header, columns
