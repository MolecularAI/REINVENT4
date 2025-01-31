import unittest

import pytest
import torch

from reinvent.models import LibinventAdapter
from reinvent.models.libinvent.models.model import DecoratorModel
from reinvent.runmodes.utils.helpers import set_torch_device


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestTokenizationWithModel(unittest.TestCase):
    def setUp(self):
        self.smiles = "c1ccccc1CC0C"

        save_dict = torch.load(self.json_config["LIBINVENT_CHEMBL_PRIOR_PATH"], map_location=self.device, weights_only=False)
        model = DecoratorModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = LibinventAdapter(model)

    def test_tokenization(self):
        tokenized = self.adapter.vocabulary.scaffold_tokenizer.tokenize(self.smiles)
        self.assertEqual(14, len(tokenized))
