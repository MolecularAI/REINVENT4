import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models import TransformerAdapter, Mol2MolModel
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.smiles_input = [ETHANE, PROPANE]
        self.smiles_output = [HEXANE, BUTANE]

        save_dict = torch.load(self.json_config["MOLFORMER_PRIOR_PATH"], weights_only=False)
        model = Mol2MolModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = TransformerAdapter(model)

        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, smiles_input, smiles_output):
        dataset = PairedDataset(
            smiles_input,
            smiles_output,
            vocabulary=self.adapter.vocabulary,
            tokenizer=SMILESTokenizer(),
        )

        dataloader = tud.DataLoader(
            dataset,
            len(dataset),
            shuffle=False,
            collate_fn=PairedDataset.collate_fn,
            generator=torch.Generator(device=self.device),
        )

        return dataloader

    def _get_src(self):
        for batch in self.data_loader:
            return batch.input.to(self.device)

    def _get_src_mask(self):
        for batch in self.data_loader:
            return batch.input_mask.to(self.device)

    def _get_trg(self):
        for batch in self.data_loader:
            return batch.output.to(self.device)

    def _get_trg_mask(self):
        for batch in self.data_loader:
            return batch.output_mask.to(self.device)

    def _get_src_shape(self):
        for batch in self.data_loader:
            return batch.input.shape

    def _get_src_mask_shape(self):
        for batch in self.data_loader:
            return batch.input_mask.shape

    def _get_trg_shape(self):
        for batch in self.data_loader:
            return batch.output.shape

    def _get_trg_mask_shape(self):
        for batch in self.data_loader:
            return batch.output_mask.shape

    def test_src_shape(self):
        result = self._get_src_shape()
        self.assertEqual(list(result), [2, 5])

    def test_src_mask_shape(self):
        result = self._get_src_mask_shape()
        self.assertEqual(list(result), [2, 1, 5])

    def test_trg_shape(self):
        result = self._get_trg_shape()
        self.assertEqual(list(result), [2, 8])

    def test_trg_mask_shape(self):
        result = self._get_trg_mask_shape()
        self.assertEqual(list(result), [2, 7, 7])

    def test_src_content(self):
        result = self._get_src()
        comparison = torch.equal(
            result, torch.tensor([[1, 60, 60, 2, 0], [1, 60, 60, 60, 2]]).to(self.device)
        )
        self.assertTrue(comparison)

    def test_src_mask_content(self):
        result = self._get_src_mask()
        comparison = torch.equal(
            result,
            torch.tensor([[[True, True, True, True, False]], [[True, True, True, True, True]]]).to(
                self.device
            ),
        )
        self.assertTrue(comparison)

    def test_trg_content(self):
        result = self._get_trg()
        comparison = torch.equal(
            result,
            torch.tensor([[1, 60, 60, 60, 60, 60, 60, 2], [1, 60, 60, 60, 60, 2, 0, 0]]).to(
                self.device
            ),
        )
        self.assertTrue(comparison)

    def test_trg_mask_content(self):
        result = self._get_trg_mask()
        comparison = torch.equal(
            result,
            torch.tensor(
                [
                    [
                        [True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False],
                        [True, True, True, False, False, False, False],
                        [True, True, True, True, False, False, False],
                        [True, True, True, True, True, False, False],
                        [True, True, True, True, True, True, False],
                        [True, True, True, True, True, True, True],
                    ],
                    [
                        [True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False],
                        [True, True, True, False, False, False, False],
                        [True, True, True, True, False, False, False],
                        [True, True, True, True, True, False, False],
                        [True, True, True, True, True, True, False],
                        [True, True, True, True, True, True, False],
                    ],
                ]
            ).to(self.device),
        )
        self.assertTrue(comparison)
