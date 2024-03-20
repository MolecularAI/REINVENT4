"""LibInvent transfer learning

Train a given model with new data.  The data comes from a file with SMILES
strings.  The file is assumed to be in multi-column format separated by commas
(CSV) or spaces.  The SMILES strings are taken from the first two columns.

The two SMILES in each row correspond to a single SMILES (scaffold) and two
pipe-symbol (|) separated SMILES fragments (the decorators) and  e.g.
[*]C#CC1(O)CC2CCC(C1)N2[*]      *c1cccc(F)c1|*C(=O)OC

The asterisks (*) are the attachment points to form a complete molecule.  The
order of the columns must follow the order in the model (file).  Currently,
this means that the scaffolds are in column 1 and the decorators in column 2.
"""

from __future__ import annotations

__all__ = ["Libinvent"]

from .learning import Learning
from reinvent.models.libinvent.models.dataset import DecoratorDataset


class Libinvent(Learning):
    """Handle LibInvent specifics"""

    def prepare_dataloader(self):
        self.dataset = DecoratorDataset(
            scaffold_decoration_smi_list=self.smilies,
            vocabulary=self.model.get_vocabulary(),
        )

        self.validation_dataset = None

        if self.validation_smilies:
            self.validation_dataset = DecoratorDataset(
                scaffold_decoration_smi_list=self.validation_smilies,
                vocabulary=self.model.get_vocabulary(),
            )

        self.collate_fn = DecoratorDataset.collate_fn

        self._common_dataloader()

    def train_epoch(self):
        return self._train_epoch_common()

    def compute_nll(self, batch):
        scaffold_batch, decorator_batch = batch
        return self.model.likelihood(*scaffold_batch, *decorator_batch)

    @staticmethod
    def collate_fn(data):
        return DecoratorDataset.collate_fn(data)
