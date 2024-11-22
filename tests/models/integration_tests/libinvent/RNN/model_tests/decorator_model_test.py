import pytest
import unittest

import torch
import torch.utils.data as tud

from reinvent.runmodes.utils.helpers import set_torch_device
from reinvent.models.model_mode_enum import ModelModeEnum
from reinvent.models.libinvent.models.dataset import Dataset
from reinvent.models.libinvent.models.model import DecoratorModel
from tests.test_data import SCAFFOLD_SUZUKI


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestDecoratorModel(unittest.TestCase):
    def setUp(self):
        input_scaffold = SCAFFOLD_SUZUKI
        scaffold_list_2 = [input_scaffold, input_scaffold]
        scaffold_list_3 = [input_scaffold, input_scaffold, input_scaffold]
        self._model_regime = ModelModeEnum()
        self._decorator = DecoratorModel.load_from_file(
            self.json_config["LIBINVENT_CHEMBL_PRIOR_PATH"],
            "inference",
            torch.device(self.device),
        )
        set_torch_device(self.device)

        dataset_2 = Dataset(
            scaffold_list_2,
            self._decorator.vocabulary.scaffold_vocabulary,
            self._decorator.vocabulary.scaffold_tokenizer,
        )
        self.dataloader_2 = tud.DataLoader(
            dataset_2, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn
        )

        dataset_3 = Dataset(
            scaffold_list_3,
            self._decorator.vocabulary.scaffold_vocabulary,
            self._decorator.vocabulary.scaffold_tokenizer,
        )
        self.dataloader_3 = tud.DataLoader(
            dataset_3, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn
        )

    def test_double_scaffold_input(self):
        for batch in self.dataloader_2:
            (
                scaffold_smiles,
                decoration_smiles,
                nlls,
            ) = self._decorator.sample_decorations(*batch)

        self.assertEqual(2, len(scaffold_smiles))
        self.assertEqual(2, len(decoration_smiles))
        self.assertEqual(2, len(nlls))

    def test_triple_scaffold_input(self):
        for batch in self.dataloader_3:
            (
                scaffold_smiles,
                decoration_smiles,
                nlls,
            ) = self._decorator.sample_decorations(*batch)

        self.assertEqual(3, len(scaffold_smiles))
        self.assertEqual(3, len(decoration_smiles))
        self.assertEqual(3, len(nlls))
