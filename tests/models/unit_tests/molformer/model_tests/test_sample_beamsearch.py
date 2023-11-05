import unittest

import torch.utils.data as tud

from reinvent.models.mol2mol.dataset.dataset import Dataset
from reinvent.models.mol2mol.enums import SamplingModesEnum
from reinvent.models.mol2mol.models.vocabulary import SMILESTokenizer
from tests.test_data import BENZENE, TOLUENE, ANILINE
from tests.models.unit_tests.molformer.fixtures import mocked_molformer_model


class TestModelSampling(unittest.TestCase):
    def setUp(self):

        self._model = mocked_molformer_model()
        self._sample_mode_enum = SamplingModesEnum()

        smiles_list = [BENZENE]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [TOLUENE, ANILINE]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [BENZENE, TOLUENE, ANILINE]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

        self.beam_size = 3
        self._model.set_beam_size(self.beam_size)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample_molecules(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.BEAMSEARCH)

    def test_single_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_1)
        self.assertEqual(3, len(smiles1))
        self.assertEqual(3, len(smiles2))
        self.assertEqual(3, len(nll))

    def test_double_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_2)
        self.assertEqual(6, len(smiles1))
        self.assertEqual(6, len(smiles2))
        self.assertEqual(6, len(nll))

    def test_triple_input(self):
        smiles1, smiles2, nll = self._sample_molecules(self.data_loader_3)
        self.assertEqual(9, len(smiles1))
        self.assertEqual(9, len(smiles2))
        self.assertEqual(9, len(nll))
