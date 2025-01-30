import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.models.transformer.pepinvent.pepinvent import PepinventModel
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_INPUT3


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestPepInventModel(unittest.TestCase):
    def setUp(self):

        save_dict = torch.load(self.json_config["PEPINVENT_PRIOR_PATH"], map_location=self.device, weights_only=False)
        self._model = PepinventModel.create_from_dict(
            save_dict, "inference", torch.device(self.device)
        )
        set_torch_device(self.device)

        self._sample_mode_enum = SamplingModesEnum()

        smiles_list = [PEPINVENT_INPUT1]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [PEPINVENT_INPUT1, PEPINVENT_INPUT2]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_INPUT3]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

        smiles_list = [PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_INPUT3]
        self.data_loader_4 = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.MULTINOMIAL)

    def test_single_input(self):
        results = self._sample(self.data_loader_1)

        self.assertEqual(1, len(results[0]))
        self.assertEqual(1, len(results[1]))
        self.assertEqual(1, len(results[2]))

    def test_double_input(self):
        results = self._sample(self.data_loader_2)

        self.assertEqual(2, len(results[0]))
        self.assertEqual(2, len(results[1]))
        self.assertEqual(2, len(results[2]))

    def test_triple_input(self):
        results = self._sample(self.data_loader_3)

        self.assertEqual(3, len(results[0]))
        self.assertEqual(3, len(results[1]))
        self.assertEqual(3, len(results[2]))
