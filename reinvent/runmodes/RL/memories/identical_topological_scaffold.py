from __future__ import annotations
from typing import List, Optional

import numpy as np

from .diversity_filter import DiversityFilter


class IdenticalTopologicalScaffold(DiversityFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray
    ) -> Optional[List]:
        """Compute the score"""

        return self.score_scaffolds(scores, smilies, mask, topological=True)
