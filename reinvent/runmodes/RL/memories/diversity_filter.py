"""The diversity filter is a memory for repeated SMILES.

Depending on the concrete filter, scaffolds or SMILES that are repeatedly found
are memorized. Filtering happens on a given minimum score.
"""

from __future__ import annotations

from datetime import timedelta
import time
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from .utils.diversity_results import DiversityResults

import numpy as np

if TYPE_CHECKING:
    from .penalty_base import BasePenalty
    from reinvent.models.model_factory.sample_batch import SampleBatch
    from reinvent.runmodes.RL.memories.intrinsic_reward_base import IntrinsicReward

logger = logging.getLogger(__name__)


class DiversityFilter(ABC):
    """Keep track of repeated SMILES and filter by a minimum score threshold."""

    def __init__(
        self,
        penalty_function: BasePenalty,
        intrinsic_reward: IntrinsicReward | None,
        bucket_size: int,
        minscore: float,
        minsimilarity: float | None,
        penalty_multiplier: float | None,
        rdkit_smiles_flags: dict,
        debug: bool = False,
        **_,
    ) -> None:
        """Set up the diversity filters.

        :param bucket_size: size of each bucket
        :param minscore: minimum score
        :param minsimilarity: minimum similarity
        :param penalty_multiplier: pnealty multiplier
        :param rdkit_smiles_flags: RDKit flags for SMILES conversion
        :param rdkit_smiles_flags: RDKit flags for canonicalization
        :param debug: whether to enable debug mode that includes metrics in the tensorboard logs, but slows down training. 
        """
        self.bucket_size = bucket_size
        self.minscore = minscore
        self.minsimilarity = minsimilarity
        self.penalty_multiplier = penalty_multiplier
        self.rdkit_smiles_flags = rdkit_smiles_flags
        self.penalty_function = penalty_function
        self.intrinsic_reward = intrinsic_reward
        self.debug = debug

        self.smiles_memory = set()

    def get_memory_size(self) -> int:
        """Get the size of the internal SMILES memory."""
        return len(self.smiles_memory)

    def purge_memories(self) -> None:
        """Purge the internal SMILES memory."""
        self.smiles_memory = set()

    @abstractmethod
    def count_full_buckets(self) -> int:
        """Count the number of full buckets."""

    @abstractmethod
    def count_total_buckets(self) -> int:
        """Count the total number of buckets."""

    @abstractmethod
    def calculate_penalty(self, scores: np.ndarray, smilies: list[str]) -> tuple[np.ndarray, None | list[tuple[list[int], int]]]:
        """Penalize the score according to the concrete filter.

        :param scores: an array with precomputed scores
        :param smilies: an array with SMILES strings
        :return: array with the penalized scores
        """

    def post_update(
        self,
        results: DiversityResults | None,
        scores: np.ndarray,
        sampled: SampleBatch,
        active_idxs: list[int],
        penalties: np.ndarray | None,
        mask: np.ndarray,
    ) -> DiversityResults | None:
        """Post-update hook ran after applying the diversity filter and calculating penalties.
        Can be used to modify the reported results and to update the internal memory.

        :param scores: an array with precomputed scores
        :param sampled: the sampled batch including SMILES and NLL values
        :param mask: mask for valid SMILES
        :return: results of the diversity filter application
        """
        return results

    def update_score(
        self,
        scores: np.ndarray,
        sampled: SampleBatch,
        mask: np.ndarray,
    ) -> DiversityResults | None:
        """Update the score according to the concrete filter.

        :param scores: an array with precomputed scores
        :param sampled: the sampled batch including SMILES and NLL values
        :param mask: mask for valid SMILES
        :return: array with the updated scores and scaffolds where available
        """
        mean_extrinsic_score = scores.mean()
        smilies = sampled.smilies
        start = time.time()

        active_idxs = []
        active_smilies = []
        results: DiversityResults | None = None
        # check if molecule has been seen before, or save it when it passes threshold
        for i in np.flatnonzero(mask):
            if smilies[i] in self.smiles_memory:
                scores[i] = 0
            elif scores[i] >= self.minscore:
                self.smiles_memory.add(smilies[i])
                active_idxs.append(i)
                active_smilies.append(smilies[i])
        active_scores = scores[active_idxs]

        if len(active_idxs) == 0:
            if self.debug:
                logger.debug("No active SMILES found")
                results = DiversityResults(
                    runtime=None,
                    memory_size=self.get_memory_size(),
                    bucket_max_size=self.bucket_size,
                    num_full_buckets=None,
                    num_total_buckets=None,
                    num_active=0,
                    mean_extrinsic_score=mean_extrinsic_score,
                )
            results = self.post_update(results, scores, sampled, active_idxs, None, mask)
            return results

        penalties, modified_bucket_occupation = self.calculate_penalty(active_scores, active_smilies)

        scores[active_idxs] *= penalties

        num_downscored = np.count_nonzero(penalties < 0.99)

        global_modified_bucket_occupation = None
        if modified_bucket_occupation is not None:
            global_modified_bucket_occupation = [
                ([active_idxs[idx] for idx in bucket_idxs], bucket_occupation)
                for bucket_idxs, bucket_occupation in modified_bucket_occupation
            ]

        # Add intrinsic reward if applicable
        if self.intrinsic_reward is not None and global_modified_bucket_occupation is not None:
            self.intrinsic_reward.add_intrinsic_reward(
                scores, sampled, active_idxs, global_modified_bucket_occupation
            )

        if self.debug:
            results = DiversityResults(
                runtime=0,
                memory_size=self.get_memory_size(),
                bucket_max_size=self.bucket_size,
                num_full_buckets=self.count_full_buckets(),
                num_total_buckets=self.count_total_buckets(),
                num_active=len(active_idxs),
                num_downscored=int(num_downscored),
                mean_extrinsic_score=mean_extrinsic_score,
                modified_bucket_occupation=global_modified_bucket_occupation,
            )

            results = self.post_update(
                results, scores, sampled, active_idxs, penalties, mask
            )

            end = time.time()
            logger.info("DF took: %s", timedelta(seconds=end - start))

            # calculate the final runtime after post-update, which can be used to update the memory
            results.runtime = end - start # type: ignore
            return results
