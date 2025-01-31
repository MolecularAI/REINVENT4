"""LinkInvent transfer learning

Train a given model with new data.  The data comes from a file with SMILES
strings.  The file is assumed to be in multi-column format separated by commas
(CSV) or spaces.  The SMILES strings are taken from the first two columns.

The two SMILES in each row correspond to two pipe-symbol (|) separated SMILES
fragments (the warheads, 'input') and a single SMILES (linker. 'target',
'output') e.g.
*C#CCO|*CCC#CCCCCCCC(C)C   [*]C#CC(O)CCCCCCC[*]

The asterisks (*) are the attachment points to form a complete molecule.  The
order of the columns must follow the order in the model (file).  Currently,
this means that the warheads/input are in column 1 and the linker/target in
column 2.  See (the model is read from the torch pickle file)

>>> import torch
>>> model = torch.load('link_invent_prior.model', weights_only=False)
>>> model['vocabulary'].input.vocabulary.tokens()
>>> model['vocabulary'].target.vocabulary.tokens()
"""

from __future__ import annotations

__all__ = ["Linkinvent"]
import logging

from .learning import Learning
from reinvent.models.linkinvent.dataset.paired_dataset import PairedDataset

logger = logging.getLogger(__name__)


class Linkinvent(Learning):
    """Handle LinkInvent specifics"""

    def prepare_dataloader(self):
        self.dataset = PairedDataset(
            input_target_smi_list=self.smilies,
            vocabulary=self.model.get_vocabulary(),
        )

        self.validation_dataset = None

        if self.validation_smilies:
            self.validation_dataset = PairedDataset(
                input_target_smi_list=self.validation_smilies,
                vocabulary=self.model.get_vocabulary(),
            )

        self.collate_fn = PairedDataset.collate_fn

        self._common_dataloader()

    def train_epoch(self):
        return self._train_epoch_common()

    def compute_nll(self, batch):
        input_batch, target_batch = batch
        return self.model.likelihood(*input_batch, *target_batch)

    @staticmethod
    def collate_fn(data):
        return PairedDataset.collate_fn(data)
