"""The Pepinvent optimization algorithm"""

from __future__ import annotations

__all__ = ["PepinventLearning"]
import logging
from typing import TYPE_CHECKING

from .learning import Learning
from ...chemistry.amino_acids.amino_acids import construct_amino_acids_fragments

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

logger = logging.getLogger(__name__)


class PepinventLearning(Learning):
    """Pepinvent optimization"""

    def score(self):
        fragmented_amino_acids = construct_amino_acids_fragments(self.sampled.items2, self.sampled.items1,
                                                                 add_O=True, remove_cyclization_numbers=True)
        results = self.scoring_function(
            self.sampled.smilies, self.invalid_mask, self.duplicate_mask, fragmented_amino_acids
        )

        return results

    def update(self, results: ScoreResults):
        return self._update_common_transformer(results)
