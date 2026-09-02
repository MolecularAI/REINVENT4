import logging
import time
from datetime import timedelta

import numpy as np
from bblean import BitBirch, fps_from_smiles

from reinvent.models.model_factory.sample_batch import SampleBatch

from ..diversity_filter import DiversityFilter

# from bblean.merges import MergeAcceptFunction
from ..utils.bitbirch_diameter_merge import BitBirchTrackingDiameterMerge
from ..utils.diversity_results import DiversityResults

logger = logging.getLogger(__name__)


class BitBirchDiversityFilter(DiversityFilter):
    """Keep track of repeated SMILES using BitBIRCH and filter by a minimum score threshold and cluster sizes."""

    def __init__(
        self,
        discard=False,
        threshold: float = 0.65,
        branching_factor: int = 2500,
        # merge_criterion: str | MergeAcceptFunction | None = None,
        tolerance: float | None = None,
        recluster_interval: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.merge_criterion = BitBirchTrackingDiameterMerge(
            max_size=self.bucket_size,
            discard=discard,
        )
        self.recluster_interval = recluster_interval
        self.bb_tree = BitBirch(
            threshold=threshold,
            merge_criterion=self.merge_criterion,
            branching_factor=branching_factor,
            tolerance=tolerance,
        )

    def calculate_penalty(self, scores: np.ndarray, smilies: list[str]) -> tuple[np.ndarray, None | list[tuple[list[int], int]]]:
        self.merge_criterion.clear_indices()

        bb_size = self.bb_tree.num_fitted_fps

        # Calculate BitBirch fingerprints
        fps = fps_from_smiles(
            np.array(smilies, dtype=object),
            pack=True,
            n_features=2048,
            kind="rdkit",
        )

        # Insert fingerprints into BitBirch Tree
        self.bb_tree.fit(fps)

        converted_bucket_occupation = []

        penalties = np.ones_like(scores)

        # Find the buckets for the inserted fingerprints and their utilization
        bucket_occupation = self.merge_criterion.cluster_sizes
        for bb_idxs, bucket_utilization in bucket_occupation:
            # calculate how much to downscore based on utilization
            penalty = self.penalty_function(bucket_utilization)

            # find indicies based on the bucket indecies
            batch_idx = np.array(bb_idxs) - bb_size
            penalties[batch_idx] = penalty

            # Store the score indices for each bucket along with utilization
            converted_bucket_occupation.append((batch_idx.tolist(), bucket_utilization))


        return penalties, converted_bucket_occupation

    def post_update(
        self,
        results: DiversityResults | None,
        scores: np.ndarray,
        sampled: SampleBatch,
        active_idxs: list[int],
        penalties: np.ndarray | None,
        mask: np.ndarray,
    ) -> DiversityResults | None:
        # Recluster every recluster_interval steps if set
        bb_size = self.bb_tree.num_fitted_fps - len(active_idxs)

        if (
            self.recluster_interval
            and self.bb_tree.num_fitted_fps // self.recluster_interval
            > bb_size // self.recluster_interval
        ):
            self.bb_tree.set_merge("tolerance-diameter")
            logger.info("Running BitBirch Reclustering")
            self.bb_tree.recluster_inplace()
            self.bb_tree.set_merge("diameter")
            logger.info("BitBirch Reclustering Finished")
        return results
    
    def count_full_buckets(self) -> int:
        cluster_mol_ids = self.bb_tree.get_cluster_mol_ids()
        return sum(len(c) > self.bucket_size for c in cluster_mol_ids)
    
    def count_total_buckets(self) -> int:
        cluster_mol_ids = self.bb_tree.get_cluster_mol_ids()
        return len(cluster_mol_ids)

    def purge_memories(self) -> None:
        """Purge the internal scaffold and SMILES memories."""
        self.bb_tree.reset()
        super().purge_memories()
