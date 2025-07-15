import unittest

import pytest
import torch
import torch.utils.data as tud

from reinvent.models import TransformerAdapter, Mol2MolModel
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.create_adapter import compatibility_setup
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

        compatibility_setup(model)

        self.adapter = TransformerAdapter(model)
        self.vocabulary = self.adapter.vocabulary
        self.tokenizer = SMILESTokenizer()

        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, smiles_input, smiles_output):
        dataset = PairedDataset(
            smiles_input,
            smiles_output,
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
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

    def _get_content(self, smiles_list):
        result = []
        for smi in smiles_list:
            tokenized_input = self.tokenizer.tokenize(smi)
            en_input = self.vocabulary.encode(tokenized_input)
            result.append(list(en_input))

        max_length = max(len(lst) for lst in result)
        padded_result = [lst + [self.vocabulary.pad_token] * (max_length - len(lst)) for lst in result]
        return padded_result

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
            result, torch.tensor(self._get_content(self.smiles_input)).to(self.device)
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
            torch.tensor(self._get_content(self.smiles_output)).to(
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
