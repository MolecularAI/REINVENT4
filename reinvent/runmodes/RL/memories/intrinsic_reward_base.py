"""The diversity filter is a memory for repeated SMILES

Depending on the concrete filter, scaffolds or SMILES that are repeatedly found
are memorized. Filtering happens on a given minimum score.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

from reinvent.models.model_factory.sample_batch import SampleBatch

logger = logging.getLogger(__name__)


class IntrinsicReward(ABC):
    """Keep track of repeated SMILES and filter by a minimum score threshold"""

    def __init__(
        self,
        device: torch.device,
        prior_model_file_path: str,
        learning_rate: float,
    ):
        """Set up the diversity filters.
        :param device: device to use
        :param prior_model_file_path: file path to the prior model for intrinsic reward model (e.g., RND)
        :param learning_rate: learning rate for intrinsic reward model (e.g., RND)
        """

        self.device = device
        self.prior_model_file_path = prior_model_file_path
        self.learning_rate = learning_rate

    @abstractmethod
    def add_intrinsic_reward(
        self,
        scores: np.ndarray,
        sampled: SampleBatch,
        active_idxs: list[int],
        bucket_utilization: list[tuple[list[int], int]]
    ) -> np.ndarray | None:
        """Update the score according to the intrinsic reward method.

        :param scores: an array with precomputed scores
        :param sampled: batch of sampled SMILES
        :param active_idxs: indices of active SMILES
        :param bucket_utilization: list of tuples containing bucket entry indices and their utilization
        :return: array with the updated scores and scaffolds where available
        """
