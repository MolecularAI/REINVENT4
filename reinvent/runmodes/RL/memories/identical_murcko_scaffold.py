from __future__ import annotations
from typing import Optional, List

import numpy as np

from .diversity_filter import DiversityFilter


class IdenticalMurckoScaffold(DiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray
    ) -> Optional[List]:
        """Compute the score"""

        return self.score_scaffolds(scores, smilies, mask, topological=False)
