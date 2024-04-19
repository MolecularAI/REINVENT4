import pytest
import unittest

import torch

from reinvent.models import LinkinventAdapter, SampledSequencesDTO
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE
from tests.models.unit_tests.linkinvent.RNN.fixtures import mocked_linkinvent_model


@pytest.mark.usefixtures("device")
class TestLinkInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        device = torch.device(self.device)
        linkinvent_model = mocked_linkinvent_model()
        linkinvent_model.network.to(device)
        linkinvent_model.device = device

        set_torch_device(device)
        self._model = LinkinventAdapter(linkinvent_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
