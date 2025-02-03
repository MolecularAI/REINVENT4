import unittest

import pytest
import torch

from reinvent.models import TransformerAdapter, SampledSequencesDTO, Mol2MolModel
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        save_dict = torch.load(self.json_config["MOLFORMER_PRIOR_PATH"], weights_only=False)
        model = Mol2MolModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = TransformerAdapter(model)

    def test_len_likelihood_smiles(self):
        results = self.adapter.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
