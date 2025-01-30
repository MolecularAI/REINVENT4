import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models import TransformerAdapter, Mol2MolModel
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import METAMIZOLE, COCAINE, AMOXAPINE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestModelSampling(unittest.TestCase):
    def setUp(self):

        save_dict = torch.load(self.json_config["MOLFORMER_PRIOR_PATH"], weights_only=False)
        model = Mol2MolModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = TransformerAdapter(model)

        smiles_list = [METAMIZOLE]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [METAMIZOLE, COCAINE]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [METAMIZOLE, COCAINE, AMOXAPINE]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self.adapter.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )
        return dataloader

    def _sample_molecules(self, data_loader):
        results = []
        for batch in data_loader:
            src, src_mask = batch
            for sampled_sequence in self.adapter.sample(
                src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL
            ):
                results.append(sampled_sequence.output)
        return results

    def test_single_input(self):
        results = self._sample_molecules(self.data_loader_1)
        self.assertEqual(1, len(results))

    def test_double_input(self):
        results = self._sample_molecules(self.data_loader_2)
        self.assertEqual(2, len(results))

    def test_triple_input(self):
        results = self._sample_molecules(self.data_loader_3)
        self.assertEqual(3, len(results))
