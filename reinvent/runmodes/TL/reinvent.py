"""Reinvent transfer learning

Train a given model with new data.  The data comes from a file with SMILES
strings.  The file is assumed to be in multi-column format separated by commas
(CSV) or spaces.  The SMILES string is extracted from the first column.

The SMILES strings is expected to be a complete molecule.
"""

from __future__ import annotations

__all__ = ["Reinvent"]

from .learning import Learning
from reinvent.models.reinvent.models.dataset import Dataset
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer


class Reinvent(Learning):
    """Handle Reinvent specifics"""

    def prepare_dataloader(self):
        self.dataset = Dataset(
            smiles_list=self.smilies,
            vocabulary=self.model.vocabulary,
            tokenizer=SMILESTokenizer(),
            randomize=self.randomize_all_smiles,
        )

        self.validation_dataset = None

        if self.validation_smilies:
            self.validation_dataset = Dataset(
                smiles_list=self.validation_smilies,
                vocabulary=self.model.vocabulary,
                tokenizer=SMILESTokenizer(),
                randomize=self.randomize_all_smiles,  # if true much shallower minimum
            )

        self.collate_fn = Dataset.collate_fn

        self._common_dataloader()

    def train_epoch(self):
        return self._train_epoch_common()

    def compute_nll(self, batch):
        return self.model.likelihood(batch)

    @staticmethod
    def collate_fn(data):
        return Dataset.collate_fn(data)
