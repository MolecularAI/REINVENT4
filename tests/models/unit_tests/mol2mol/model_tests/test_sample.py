import pytest
import unittest

import torch
import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from tests.test_data import BENZENE, TOLUENE, ANILINE
from tests.models.unit_tests.mol2mol.fixtures import mocked_mol2mol_model


@pytest.mark.usefixtures("device")
class TestModelSampling(unittest.TestCase):
    def setUp(self):

        device = torch.device(self.device)
        self._model = mocked_mol2mol_model()
        self._model.network.to(device)
        self._model.device = device
        self._sample_mode_enum = SamplingModesEnum()

        smiles_list = [BENZENE]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [TOLUENE, ANILINE]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [BENZENE, TOLUENE, ANILINE]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample_molecules(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.MULTINOMIAL)

    def test_single_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_1)
        self.assertEqual(1, len(smiles1))
        self.assertEqual(1, len(smiles2))
        self.assertEqual(1, len(nll))

    def test_double_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_2)
        self.assertEqual(2, len(smiles1))
        self.assertEqual(2, len(smiles2))
        self.assertEqual(2, len(nll))

    def test_triple_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_3)
        self.assertEqual(3, len(smiles1))
        self.assertEqual(3, len(smiles2))
        self.assertEqual(3, len(nll))
