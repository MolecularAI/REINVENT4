from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
from reinvent.models.transformer.mol2mol.dataset.preprocessing import scaffold

from .diversity_filter import DiversityFilter


class IdenticalTopologicalScaffold(DiversityFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray, dummy
    ) -> Tuple[List, np.ndarray]:
        """Compute the score"""

        scaffolds, original_scores, _ = self.score_scaffolds(
            scores, smilies, mask, topological=True
        )

        return scaffolds, original_scores
