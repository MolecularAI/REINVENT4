import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models import LinkinventAdapter
from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.linkinvent.link_invent_model import LinkInventModel
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import WARHEAD_PAIR


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestLinkInventModel(unittest.TestCase):
    def setUp(self):

        self.smiles = WARHEAD_PAIR

        save_dict = torch.load(
            self.json_config["LINKINVENT_CHEMBL_PRIOR_PATH"], map_location=self.device
        )
        model = LinkInventModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = LinkinventAdapter(model)

        ds1 = Dataset([self.smiles], self.adapter.vocabulary.input)
        self.dataloader_1 = tud.DataLoader(
            ds1, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn
        )

        ds2 = Dataset([self.smiles] * 2, self.adapter.vocabulary.input)
        self.dataloader_2 = tud.DataLoader(
            ds2, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn
        )

        ds3 = Dataset([self.smiles] * 3, self.adapter.vocabulary.input)
        self.dataloader_3 = tud.DataLoader(
            ds3, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn
        )

    def _sample_linker(self, data_loader):
        for batch in data_loader:
            return self.adapter.sample(*batch)

    def test_single_warhead_input(self):
        results = self._sample_linker(self.dataloader_1)

        self.assertEqual(1, len(results.output))

    def test_two_warheads_input(self):
        results = self._sample_linker(self.dataloader_2)

        self.assertEqual(2, len(results.output))

    def test_three_warheads_input(self):
        results = self._sample_linker(self.dataloader_3)

        self.assertEqual(3, len(results.output))
