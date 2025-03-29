"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np

from reinvent.runmodes.utils import make_grid_image

if TYPE_CHECKING:
    from reinvent.runmodes.RL.reports import RLReportData

logger = logging.getLogger(__name__)

ROWS = 5
COLUMNS = 6


class RLTBReporter:
    """Tensorboard class"""

    def __init__(self, reporter):
        self.reporter = reporter

    def submit(self, data: RLReportData) -> None:
        """Write out TensorBoard data

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
            self.reporter.add_scalar(name, np.nanmean(_scores[mask_idx]), step)

        for name, _scores in zip(names, raw_scores):
            if _scores.dtype.char == "U":  # raw scores may contain strings
                continue

            self.reporter.add_scalar(f"{name} (raw)", np.nanmean(_scores[mask_idx]), step)

        self.reporter.add_scalar(f"Loss", data.loss, step)

        # NOTE: for some reason this breaks on Windows because the necessary
        #       subdirectory cannot be created
        self.reporter.add_scalars(
            "Loss (likelihood averages)",
            {
                "prior NLL": data.prior_mean_nll,
                "agent NLL": data.agent_mean_nll,
                "augmented NLL": data.augmented_mean_nll,
            },
            step,
        )

        self.reporter.add_scalar("Fraction of valid SMILES", data.fraction_valid_smiles, step)
        self.reporter.add_scalar(
            "Fraction of duplicate SMILES", data.fraction_duplicate_smiles, step
        )
        self.reporter.add_scalar("Average total score", data.mean_score, step)

        if data.bucket_max_size:
            self.reporter.add_scalar(
                f"Number of scaffolds found more than {data.bucket_max_size} times",
                data.num_full_buckets,
                step,
            )
            self.reporter.add_scalar("Number of unique scaffolds", data.num_total_buckets, step)

        labels = [f"score={score:.2f}" for score in results.total_scores]
        sample_size = ROWS * COLUMNS

        image_tensor = make_grid_image(data.smilies, labels, sample_size, ROWS)

        if image_tensor is not None:
            self.reporter.add_image(
                f"First {sample_size} Structures", image_tensor, step, dataformats="CHW"
            )  # channel, height, width

        if data.isim:
            self.reporter.add_scalar(f"iSIM: Average similarity", data.isim, step)
