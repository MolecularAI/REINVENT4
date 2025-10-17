from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from .diversity_filter import DiversityFilter


class IdenticalMurckoScaffold(DiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray, dummy
    ) -> Tuple[List, np.ndarray]:
        """Compute the score"""

        scaffolds, original_scores, _ = self.score_scaffolds(
            scores, smilies, mask, topological=False
        )

        return scaffolds, original_scores
