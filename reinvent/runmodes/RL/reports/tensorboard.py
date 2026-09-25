"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np
import psutil
import os

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

        layout = {
            "Loss (likelihood averages)": {
                "Loss (likelihood averages)": [
                    "Multiline",
                    ["NLL/prior", "NLL/agent", "NLL/augmented"],
                ]
            }
        }
        self.reporter.add_custom_scalars(layout)

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

            for original_scores in np.array(
                transformed_result.component_result.fetch_scores(results.smilies, transpose=True)
            ):
                raw_scores.append(original_scores)

        for name, _scores in zip(names, scores):
            self.reporter.add_scalar(name, np.nanmean(_scores[mask_idx]), step)

        for name, _scores in zip(names, raw_scores):
            if _scores.dtype.char == "U":  # raw scores may contain strings
                continue

            self.reporter.add_scalar(f"{name} (raw)", np.nanmean(_scores[mask_idx]), step)

        self.reporter.add_scalar(f"Loss", data.loss, step)

        self.reporter.add_scalar("NLL/prior", data.prior_mean_nll, step)
        self.reporter.add_scalar("NLL/agent", data.agent_mean_nll, step)
        self.reporter.add_scalar("NLL/augmented", data.augmented_mean_nll, step)

        self.reporter.add_scalar("Fraction of valid SMILES", data.fraction_valid_smiles, step)
        self.reporter.add_scalar(
            "Fraction of duplicate SMILES", data.fraction_duplicate_smiles, step
        )
        self.reporter.add_scalar("Average total score", data.mean_score, step)

        if data.diversity_results:
            ds = data.diversity_results
            if ds.num_full_buckets is not None:
                self.reporter.add_scalar(
                    f"Diversity/Number of buckets with more than {ds.bucket_max_size}",
                    ds.num_full_buckets,
                    step,
                )
            if ds.num_active is not None:
                self.reporter.add_scalar("Diversity/Number active molecules", ds.num_active, step)
            if ds.num_downscored is not None:
                self.reporter.add_scalar("Diversity/Number downscored molecules", ds.num_downscored, step)
            if ds.num_total_buckets is not None:
                self.reporter.add_scalar("Diversity/Number of unique buckets", ds.num_total_buckets, step)
            if ds.memory_size is not None:
                self.reporter.add_scalar("Diversity/Memory Size", ds.memory_size, step)
            if ds.runtime is not None:
                self.reporter.add_scalar("Diversity/Runtime", ds.runtime, step)
            if ds.intrinsic_reward is not None: 
                self.reporter.add_scalar("Diversity/Intrinsic Reward added per Batch", ds.intrinsic_reward, step)
            if ds.mean_extrinsic_score is not None: 
                self.reporter.add_scalar("Diversity/Average Extrinsic Score", ds.mean_extrinsic_score, step)

            if ds.modified_bucket_occupation is not None:
                num_modified_buckets = len(ds.modified_bucket_occupation)
                avg_modified_bucket_occupation = (
                    sum(v[1] for v in  ds.modified_bucket_occupation) / num_modified_buckets
                    if num_modified_buckets > 0
                    else 0
                )
                self.reporter.add_scalar(
                    "Diversity/Modified Buckets", num_modified_buckets, step
                )
                self.reporter.add_scalar(
                    "Diversity/Average Modified Bucket Occupation", avg_modified_bucket_occupation, step
                )

        if data.inception_results:
            ir = data.inception_results
            if ir.memory_size is not None:
                self.reporter.add_scalar("Inception/Memory Size", ir.memory_size, step)
            if ir.num_new_added is not None:
                self.reporter.add_scalar("Inception/New Added", ir.num_new_added, step)
            if ir.num_evicted is not None:
                self.reporter.add_scalar("Inception/Evicted", ir.num_evicted, step)
            if ir.top_score is not None:
                self.reporter.add_scalar("Inception/Top Score", ir.top_score, step)
            if ir.mean_score is not None:
                self.reporter.add_scalar("Inception/Mean Score", ir.mean_score, step)
            if ir.bottom_score is not None:
                self.reporter.add_scalar("Inception/Bottom Score", ir.bottom_score, step)
            if ir.mean_sampled_score is not None:
                self.reporter.add_scalar("Inception/Mean Sampled Score", ir.mean_sampled_score, step)
            if ir.is_weight_max is not None:
                self.reporter.add_scalar("Inception/IS Weight Max", ir.is_weight_max, step)
            if ir.is_weight_entropy is not None:
                self.reporter.add_scalar("Inception/IS Weight Entropy", ir.is_weight_entropy, step)
            if ir.mean_agent_ll_drift is not None:
                self.reporter.add_scalar("Inception/Mean Agent LL Drift", ir.mean_agent_ll_drift, step)
            if ir.runtime is not None:
                self.reporter.add_scalar("Inception/Runtime", ir.runtime, step)

        # Process Memory every 20 steps
        if step % 20 == 0:
            # Get the ID of the current process
            process = psutil.Process(os.getpid())
            # Get memory info in bytes
            mem_info = process.memory_info()
            # Convert to Gibabytes (GB)
            rss_mb = 0.001 * mem_info.rss / 1024 ** 2
            vms_mb = 0.001 * mem_info.vms / 1024 ** 2
            self.reporter.add_scalar("Physical Memory (GB)", rss_mb, step)
            self.reporter.add_scalar("Virtual Memory (GB)", vms_mb, step)

        labels = [f"score={score:.2f}" for score in results.total_scores]
        sample_size = ROWS * COLUMNS

        image_tensor = make_grid_image(results.smilies, labels, sample_size, ROWS)

        if image_tensor is not None:
            self.reporter.add_image(
                f"First {sample_size} Structures", image_tensor, step, dataformats="CHW"
            )  # channel, height, width

        if data.isim:
            self.reporter.add_scalar(f"iSIM: Average similarity", data.isim, step)
