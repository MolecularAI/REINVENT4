import logging
from collections.abc import Sequence

import numpy as np

from reinvent.models.model_factory.sample_batch import SampleBatch

from ..intrinsic_reward_base import IntrinsicReward

logger = logging.getLogger(__name__)


class InformationReward(IntrinsicReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_intrinsic_reward(
        self,
        scores: np.ndarray,
        sampled: SampleBatch,
        active_idxs: list[int],
        bucket_utilization: list[tuple[list[int], int]],
    ) -> np.ndarray | None:
        if len(bucket_utilization) == 0:
            return

        # Size to full scores so that any score-space index from the filters
        # can be used directly (scaffold filter stores score indices; bitbirch
        # filter converts tree indices to score indices before this call).
        all_entropy = np.zeros(len(scores))

        n_buckets = len(bucket_utilization)

        for idxs, bucket_occupation in bucket_utilization:
            all_entropy[idxs] = -np.log(bucket_occupation / (n_buckets + 1e-6))

        # Normalize information gains over the active molecules only
        active_entropy = all_entropy[active_idxs]
        if len(active_entropy) > 2:
            active_entropy = (active_entropy - np.amin(active_entropy)) / (
                np.amax(active_entropy) - np.amin(active_entropy) + 1e-6
            )
            all_entropy[active_idxs] = active_entropy

        scores[active_idxs] += all_entropy[active_idxs]

        return all_entropy[active_idxs]
