import pytest
import unittest

import torch

from reinvent.models import LibinventAdapter, SampledSequencesDTO
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE
from tests.models.unit_tests.libinvent.RNN.fixtures import mocked_decorator_model


@pytest.mark.usefixtures("device")
class TestLibInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        device = torch.device(self.device)
        decoder_model = mocked_decorator_model()
        decoder_model.network.to(device)
        decoder_model.device = device

        set_torch_device(device)

        self._model = LibinventAdapter(decoder_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
