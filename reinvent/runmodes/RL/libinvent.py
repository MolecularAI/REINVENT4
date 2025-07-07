"""The LibInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LibinventLearning", "LibinventTransformerLearning"]
from typing import TYPE_CHECKING

import numpy as np

from .learning import Learning
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent_plugins.normalizers.rdkit_smiles import normalize

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


class LibinventLearning(Learning):
    """LibInvent optimization"""

    def update(self, results: ScoreResults):
        if self.prior.version == 1:  # RNN-based
            return self._update_common(results)
        elif self.prior.version == 2:  # Transformer-based
            return self._update_common_transformer(results)

    def score(self):
        """Compute the score for the SMILES strings.

        Overwrites generic method to pass on annotated smiles for use in reaction filter
        """

        connectivity_annotated_smiles = self.sampled.smilies

        results = self.scoring_function(
            normalize(self.sampled.smilies, keep_all=True),
            self.invalid_mask,
            self.duplicate_mask,
            connectivity_annotated_smiles=connectivity_annotated_smiles,
        )

        return results


LibinventTransformerLearning = LibinventLearning
