import logging
import time

import numpy as np

from reinvent.models.model_factory.sample_batch import SampleBatch

# from bblean.merges import MergeAcceptFunction
from ..memories.utils.diversity_results import DiversityResults
from .intrinsic_penalty import IntrinsicPenalty

logger = logging.getLogger(__name__)

class InformationPenalty(IntrinsicPenalty):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_score(
        self,
        scores: np.ndarray,
        smilies: list[str],
        mask: np.ndarray, sampled: SampleBatch
    ) -> DiversityResults | None:
        mean_extrinsic_score = scores.mean()
        start = time.time()
        scaffolds, hits_idxs = self.score_scaffolds(scores, smilies, mask, topological=False)

        # Only use scaffolds of hits
        hit_scaffolds = [s for i,s in enumerate(scaffolds) if i in hits_idxs]

        if not hit_scaffolds or len(hit_scaffolds) == 0:
            return

        scaff_entropy = np.zeros(len(hit_scaffolds))
        n_in_scaffold = np.zeros(len(hit_scaffolds))

        n_scaffolds = len(hit_scaffolds)

        for i_smi in range(n_scaffolds):

            # Invalid SMILES does not have score larger than 0
            scaff = hit_scaffolds[i_smi]
            n_scaff = self.scaffold_memory.bucket_count(scaff)

            scaff_entropy[i_smi] = -np.log(n_scaff / (n_scaffolds + 1e-6))

            n_in_scaffold[i_smi] = n_scaff

        # Normalize informations gains
        if len(scaff_entropy) > 2:
            scaff_entropy = (scaff_entropy - np.amin(scaff_entropy)) / (
                np.amax(scaff_entropy) - np.amin(scaff_entropy) + 1e-6
            )

        scores[hits_idxs] += scaff_entropy

        end = time.time()

        return DiversityResults(
            runtime=end-start,
            scaffolds=hit_scaffolds,
            memory_size=len(self.smiles_memory),
            bucket_max_size=self.bucket_size,
            num_full_buckets=self.scaffold_memory.count_full(),
            num_total_buckets=len(self.scaffold_memory),
            num_active=len(hits_idxs),
            intrinsic_reward=float(np.sum(scaff_entropy)),
            mean_extrinsic_score = mean_extrinsic_score,
        )
