from __future__ import annotations
from typing import List

import numpy as np

from .diversity_filter import DiversityFilter


class PenalizeSameSmiles(DiversityFilter):
    """Penalize previously generated compounds."""

    def update_score(self, scores: np.ndarray, smilies: List[str], mask: np.ndarray) -> None:
        """Compute the score"""

        for i in np.nonzero(mask)[0]:
            if smilies[i] in self.smiles_memory:
                scores[i] *= self.penalty_multiplier

            self.smiles_memory.add(smilies[i])

        return None
