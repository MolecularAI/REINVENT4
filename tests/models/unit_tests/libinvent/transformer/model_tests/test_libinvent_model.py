import pytest
import unittest

import torch
import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.utils import set_torch_device
from tests.models.unit_tests.libinvent.transformer.fixtures import mocked_libinvent_model
from tests.test_data import SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, SCAFFOLD_QUADRUPLE_POINT


@pytest.mark.usefixtures("device")
class TestLibInventModel(unittest.TestCase):
    def setUp(self):

        device = torch.device(self.device)
        self._model = mocked_libinvent_model()
        self._model.network.to(device)
        self._model.device = device
        self._sample_mode_enum = SamplingModesEnum()

        set_torch_device(device)

        smiles_list = [SCAFFOLD_SINGLE_POINT]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

        smiles_list = [SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, SCAFFOLD_QUADRUPLE_POINT]
        self.data_loader_4 = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample_decorations(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.MULTINOMIAL)

    def test_single_attachment_input(self):
        results = self._sample_decorations(self.data_loader_1)

        self.assertEqual(1, len(results[0]))
        self.assertEqual(1, len(results[1]))
        self.assertEqual(1, len(results[2]))

    def test_double_attachment_input(self):
        results = self._sample_decorations(self.data_loader_2)

        self.assertEqual(2, len(results[0]))
        self.assertEqual(2, len(results[1]))
        self.assertEqual(2, len(results[2]))

    def test_triple_attachment_input(self):
        results = self._sample_decorations(self.data_loader_3)

        self.assertEqual(3, len(results[0]))
        self.assertEqual(3, len(results[1]))
        self.assertEqual(3, len(results[2]))

    def test_quadruple_attachment_input(self):
        results = self._sample_decorations(self.data_loader_4)

        self.assertEqual(4, len(results[0]))
        self.assertEqual(4, len(results[1]))
        self.assertEqual(4, len(results[2]))
