"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import logging

import numpy as np

from reinvent.runmodes.utils import make_grid_image

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


logger = logging.getLogger(__name__)

ROWS = 5
COLUMNS = 6


@dataclass
class TBData:
    step: int
    score_results: ScoreResults
    smilies: list
    prior_nll: float
    agent_nll: float
    augmented_nll: float
    loss: float
    fraction_valid_smiles: float
    fraction_duplicate_smiles: float
    bucket_max_size: int
    num_full_buckets: int
    num_total_buckets: int
    mean_score: float
    mask_idx: np.ndarray


def write_report(reporter, data: TBData) -> None:
    """Write out TensorBoard data

    :param reporter: TB reporter for writing out the data
    :param data: data to be written out
    """

    mask_idx = data.mask_idx
    step = data.step

    results = data.score_results
    names = []
    scores = []
    raw_scores = []

    for transformed_result in results.completed_components:
        names.extend(transformed_result.component_names)

        for transformed_scores in transformed_result.transformed_scores:
            scores.append(transformed_scores)

        for original_scores in transformed_result.component_result.scores:
            raw_scores.append(original_scores)

    for name, _scores in zip(names, scores):
        reporter.add_scalar(name, np.nanmean(_scores[mask_idx]), step)

    for name, _scores in zip(names, raw_scores):
        if _scores.dtype.char == "U":  # raw scores may contain strings
            continue

        reporter.add_scalar(f"{name} (raw)", np.nanmean(_scores[mask_idx]), step)

    reporter.add_scalar(f"Loss", data.loss, step)

    # NOTE: for some reason this breaks on Windows because the necessary
    #       subdirectory cannot be created
    reporter.add_scalars(
        "Loss (likelihood averages)",
        {
            "prior NLL": data.prior_nll,
            "agent NLL": data.agent_nll,
            "augmented NLL": data.augmented_nll,
        },
        step,
    )

    reporter.add_scalar("Fraction of valid SMILES", data.fraction_valid_smiles, step)
    reporter.add_scalar("Fraction of duplicate SMILES", data.fraction_duplicate_smiles, step)
    reporter.add_scalar("Average total score", data.mean_score, step)

    if data.bucket_max_size:
        reporter.add_scalar(
            f"Number of scaffolds found more than {data.bucket_max_size} times",
            data.num_full_buckets,
            step,
        )
        reporter.add_scalar("Number of unique scaffolds", data.num_total_buckets, step)

    sample_size = ROWS * COLUMNS

    image_tensor, _ = make_grid_image(data.smilies, results.total_scores, "score", sample_size, ROWS)

    if image_tensor is not None:
        reporter.add_image(
            f"First {sample_size} Structures", image_tensor, step, dataformats="CHW"
        )  # channel, height, width
