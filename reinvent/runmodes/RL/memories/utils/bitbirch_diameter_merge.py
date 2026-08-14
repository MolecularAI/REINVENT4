from collections.abc import Sequence

from bblean.merges import DiameterMerge, DiscardSubcluster


class BitBirchTrackingDiameterMerge(DiameterMerge):
    """Tracks and optionally discards nominees once a cluster meets a size threshold.

    Args:
        max_size (int): Cluster size at or above which nominees are marked redundant.
        discard (bool): If True, raise DiscardSubcluster to exclude redundant nominees.

    Attributes:
        max_size (int): Redundancy threshold.
        cluster_sizes (list[(list[int], int)]): Stores the indicies for every inserted nominees for each cluster, along with its total size
        discard (bool): Whether to exclude redundant nominees from the final tree.
    """

    def __init__(self, max_size: int, discard: bool = False) -> None:
        """Initialize with threshold and discard policy.

        Args:
            max_size (int): Size threshold for redundancy.
            discard (bool): Exclude redundant nominees if True.
        """
        self.max_size = max_size
        self.cluster_sizes: list[tuple[Sequence[int], int]] = list()
        self.discard = discard


    # The `on_check_merge_end` hook is called after a merge check is performed
    def on_check_merge_end(self, accepted, old_idxs, nominee_idxs : Sequence[int]):
        """Post-merge hook: flag nominees as redundant and optionally discard them.

        If the merge is accepted and len(old_idxs) >= max_size, nominee indices are
        appended to redundant_mol_indices. If discard is True, DiscardSubcluster is raised.

        Args:
            accepted (bool): Whether the merge was accepted.
            old_idxs (Sequence[int]): Existing cluster indices.
            nominee_idxs (Sequence[int]): Proposed indices to add.

        Raises:
            DiscardSubcluster: When discarding redundant nominees is enabled.
        """
        # If the merge is accepted, and the size of the cluster in the tree exceeds some
        # user-defined size, tag the molecule index as "redundant"
        if accepted:
            self.cluster_sizes.append((nominee_idxs,  len(old_idxs)))
            if self.discard and len(old_idxs) >= self.max_size:
                raise DiscardSubcluster
    def clear_indices(self):
        """
        Clears the reduntant indicies
        """
        self.cluster_sizes = list()
