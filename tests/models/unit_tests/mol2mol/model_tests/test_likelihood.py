import unittest

import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE
from tests.models.unit_tests.mol2mol.fixtures import mocked_mol2mol_model


class TestLikelihood(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles_input = [ETHANE, PROPANE]
        self.smiles_output = [HEXANE, BUTANE]

        self._model = mocked_mol2mol_model()
        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, input, output):
        dataset = PairedDataset(
            input,
            output,
            vocabulary=self._model.vocabulary,
            tokenizer=SMILESTokenizer(),
        )
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn
        )
        return dataloader

    def test_len_likelihood(self):
        for batch in self.data_loader:
            results = self._model.likelihood(
                batch.input, batch.input_mask, batch.output, batch.output_mask
            )
            self.assertEqual([2], list(results.shape))
