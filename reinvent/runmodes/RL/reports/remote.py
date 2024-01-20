"""Send information to a remote server"""

from __future__ import annotations

import time
from typing import List, TYPE_CHECKING
from dataclasses import dataclass
import logging

import numpy as np

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

logger = logging.getLogger(__name__)


@dataclass
class RemoteData:
    step: int
    score_results: ScoreResults
    prior_nll: float
    agent_nll: float
    fraction_valid_smiles: float
    number_of_smiles: int
    start_time: float
    n_steps: int
    mean_score: float
    mask_idx: np.ndarray


def send_report(data: RemoteData, reporter) -> None:
    """Send data to a remote endpoint

    :param data: data to be send and transformed into JSON format
    """

    mask_idx = data.mask_idx
    step = data.step

    score_components = score_summary(data.score_results, mask_idx)
    score_components["total_score"] = float(data.mean_score)

    learning_curves = {
        "prior NLL": float(data.prior_nll),
        "agent NLL": float(data.agent_nll),
    }

    smarts_pattern = ""  # get_matching_substructure(data.score_results)
    smiles_legend_pairs = get_smiles_legend_pairs(
        np.array(data.score_results.smilies)[mask_idx],
        data.score_results.total_scores[mask_idx]
    )

    time_estimation = estimate_run_time(data.start_time, data.n_steps, step)

    record = {
        "step": step,
        "timestamp": time.time(),  # gives microsecond resolution on Linux
        "components": score_components,
        "learning": learning_curves,
        "time_estimation": time_estimation,
        "fraction_valid_smiles": float(data.fraction_valid_smiles),
        "smiles_report": {
            "smarts_pattern": smarts_pattern,
            "smiles_legend_pairs": smiles_legend_pairs,
        },
        "collected smiles in memory": data.number_of_smiles if data.number_of_smiles else 0,
    }

    reporter.send(record)


def score_summary(results: ScoreResults, mask_idx: np.ndarray) -> dict:
    """Extract the score results from ScoreResults

    :param results: results dataclass
    :param mask_idx: mask to extract only wanted results
    :returns: dictionary with scoring results
    """

    names = []
    scores = []
    raw_scores = []
    score_components = {}

    for transformed_result in results.completed_components:
        prefix = f"{transformed_result.component_type}"
        names.extend([f"{prefix}:{name}" for name in transformed_result.component_names])

        for transformed_scores in transformed_result.transformed_scores:
            scores.append(transformed_scores)

        for original_scores in transformed_result.component_result.scores:
            if original_scores.dtype.char != "U":  # exclude categorical values
                raw_scores.append(original_scores)

    for name, _scores in zip(names, scores):
        score_components[name] = np.nanmean(_scores[mask_idx]).astype(float)

    for name, _scores in zip(names, raw_scores):
        score_components[name + " (raw)"] = np.nanmean(_scores[mask_idx]).astype(float)

    return score_components


def get_smiles_legend_pairs(smilies: List[str], scores: List[str]) -> List:
    """Use the score to create a legend for each SMILES:

    :param smilies: SMILES list
    :param scores: scores list which must be of same length as smilies
    :returns: list with legend pairs
    """

    combined = [(smiles, float(score)) for smiles, score in zip(smilies, scores)]
    pairs = sorted(combined, key=lambda item: item[1], reverse=True)

    smiles_legend_pairs = [
        {"smiles": smiles, "legend": f"{score:.2f}"} for smiles, score in pairs
    ]

    return smiles_legend_pairs


def estimate_run_time(start_time, n_steps, step):
    # FIXME: The problem is that we do not know how many steps will be
    #        run. The assumption here is that all n_steps will be.
    time_elapsed = time.time() - start_time
    time_left = time_elapsed * ((n_steps - step) / step)
    summary = {"elapsed": time_elapsed, "max left": time_left}

    return summary
