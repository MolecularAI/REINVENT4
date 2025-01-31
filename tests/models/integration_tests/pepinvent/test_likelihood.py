import unittest

import pytest
import torch

from reinvent.models import PepinventAdapter, SampledSequencesDTO
from reinvent.models.transformer.pepinvent.pepinvent import PepinventModel
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import PEPINVENT_INPUT1, PEPINVENT_OUTPUT1, PEPINVENT_INPUT2, PEPINVENT_OUTPUT2


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestPepInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(PEPINVENT_INPUT1, PEPINVENT_OUTPUT1, 0.9)
        dto2 = SampledSequencesDTO(PEPINVENT_INPUT2, PEPINVENT_OUTPUT2, 0.1)
        self.sampled_sequence_list = [dto1, dto2]
        save_dict = torch.load(self.json_config["PEPINVENT_PRIOR_PATH"], map_location=self.device, weights_only=False)
        model = PepinventModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = PepinventAdapter(model)

    def test_len_likelihood_smiles(self):
        results = self.adapter.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
