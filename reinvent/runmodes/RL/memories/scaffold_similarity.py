from __future__ import annotations
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from .diversity_filter import DiversityFilter


class ScaffoldSimilarity(DiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaffold_fingerprints = {}

    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray
    ) -> Optional[List]:
        """Compute the score"""

        return self.score_scaffolds(scores, smilies, mask, topological=False, similar=True)

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
            fp = Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(scaffold))

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
