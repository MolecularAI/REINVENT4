import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models import TransformerAdapter, Mol2MolModel
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestLikelihood(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles_input = [ETHANE, PROPANE]
        self.smiles_output = [HEXANE, BUTANE]

        save_dict = torch.load(self.json_config["MOLFORMER_PRIOR_PATH"], weights_only=False)
        model = Mol2MolModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = TransformerAdapter(model)

        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, input, output):
        dataset = PairedDataset(
            input,
            output,
            vocabulary=self.adapter.vocabulary,
            tokenizer=SMILESTokenizer(),
        )
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn
        )
        return dataloader

    def test_len_likelihood(self):
        for batch in self.data_loader:
            results = self.adapter.likelihood(
                batch.input, batch.input_mask, batch.output, batch.output_mask
            )
            self.assertEqual([2], list(results.shape))
