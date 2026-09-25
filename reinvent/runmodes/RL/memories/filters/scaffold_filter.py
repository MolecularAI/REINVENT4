import logging

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Scaffolds import MurckoScaffold

from reinvent.chemistry import conversions
from reinvent.models.model_factory.sample_batch import SampleBatch

from ..diversity_filter import DiversityFilter
from ..utils.bucket_counter import BucketCounter
from ..utils.diversity_results import DiversityResults

logger = logging.getLogger(__name__)


class ScaffoldDiversityFilter(DiversityFilter):
    """Keep track of repeated SMILES and filter by a minimum score threshold and scaffolds"""

    def __init__(
        self,
        *args,
        topological: bool = False,
        similar: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topological = topological
        self.similar = similar
        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.scaffold_fingerprints = {}
        self._latest_active_scaffolds: list[str | None] = []

    def calculate_penalty(
        self, scores: np.ndarray, smilies: list[str]
    ) -> tuple[np.ndarray, None | list[tuple[list[int], int]]]:
        """Calculate penalties based on scaffold bucket utilization.

        The returned bucket indices are local to the active batch and are
        converted to score-space indices by the base class.
        """
        logger.debug("%s: active_smilies=%d", __name__, len(smilies))

        penalties = np.ones_like(scores)
        self._latest_active_scaffolds = [None] * len(smilies)
        bucket_assignments: dict[str, list[int]] = {}
        bucket_sizes: dict[str, int] = {}

        for i, smiles in enumerate(smilies):
            scaffold = self._calculate_scaffold(smiles, self.topological)

            if self.similar:
                scaffold = self._find_similar_scaffold(scaffold)

            self._latest_active_scaffolds[i] = scaffold
            self.scaffold_memory.add(scaffold)
            bucket_assignments.setdefault(scaffold, []).append(i)
            bucket_sizes[scaffold] = self.scaffold_memory.bucket_count(scaffold)

            penalties[i] = self.penalty_function(bucket_sizes[scaffold])

        bucket_occupation: list[tuple[list[int], int]] = [
            (idxs, bucket_sizes[scaffold])
            for scaffold, idxs in bucket_assignments.items()
        ]

        return penalties, bucket_occupation

    def post_update(
        self,
        results: DiversityResults | None,
        scores: np.ndarray,
        sampled: SampleBatch,
        active_idxs: list[int],
        penalties: np.ndarray | None,
        mask: np.ndarray,
    ) -> DiversityResults | None:
        scaffolds: list[str | None] = [None] * len(sampled.smilies)
        for scaffold, idx in zip(self._latest_active_scaffolds, active_idxs, strict=False):
            scaffolds[idx] = scaffold
        if results is not None:
            results.scaffolds = scaffolds
        return results

    def count_full_buckets(self) -> int:
        return self.scaffold_memory.count_full()

    def count_total_buckets(self) -> int:
        return len(self.scaffold_memory)

    def _calculate_scaffold(self, smile: str, topological: bool) -> str:
        """Compute the Murcko scaffold for the given SMILES string

        :param smile: the SMILES strings to compute the scaffold from
        :param topological: whether the scaffold should be made generic
        :returns: scaffold SMILES string
        """

        mol = conversions.smile_to_mol(smile)
        scaffold_smiles = ""

        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)

                if topological:
                    scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

                # NOTE: MolToSmiles(canonical=True) by default
                # FIXME: do not rely on default
                scaffold_smiles = conversions.mol_to_smiles(
                    scaffold, **self.rdkit_smiles_flags
                )
            except ValueError:
                pass

        return scaffold_smiles

    def _find_similar_scaffold(self, scaffold):
        """Find similar scaffolds

        Tries to find a "similar" scaffold (according to the threshold set by
        parameter "minsimilarity") and if at least one scaffold satisfies this
        criteria, it will replace the smiles' scaffold with the most similar one
        -> in effect, this reduces the number of scaffold buckets in the memory
        (the lower parameter "minsimilarity", the more pronounced the reduction)
        generate a "mol" scaffold from the smile and calculate an atom pai
         fingerprint

        :param scaffold: scaffold represented by a SMILES string
        :return: closest scaffold given a certain similarity threshold
        """

        if scaffold:
            fp = Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(scaffold)) # type: ignore

            # make a list of the stored fingerprints for similarity calculations
            fps = list(self.scaffold_fingerprints.values())

            # check, if a similar scaffold entry already exists and if so, use this one instead
            if len(fps) > 0:
                similarity_scores = DataStructs.BulkDiceSimilarity(fp, fps)
                closest = np.argmax(similarity_scores)

                if similarity_scores[closest] >= self.minsimilarity:
                    scaffold = list(self.scaffold_fingerprints.keys())[closest]
                    fp = self.scaffold_fingerprints[scaffold]

            self.scaffold_fingerprints[scaffold] = fp

        return scaffold

    def purge_memories(self):
        """Purge the internal scaffold and SMILES memories"""

        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.scaffold_fingerprints = {}
        self._latest_active_scaffolds = []

        super().purge_memories()
