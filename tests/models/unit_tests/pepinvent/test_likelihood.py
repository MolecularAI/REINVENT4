import pytest
import unittest

from reinvent.models import PepinventAdapter, SampledSequencesDTO
from tests.models.unit_tests.pepinvent.fixtures import mocked_pepinvent_model
from tests.test_data import PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_OUTPUT1, PEPINVENT_OUTPUT2


@pytest.mark.usefixtures("device")
class TestPepInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(PEPINVENT_INPUT1, PEPINVENT_OUTPUT1, 0.9)
        dto2 = SampledSequencesDTO(PEPINVENT_INPUT2, PEPINVENT_OUTPUT2, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        pepinvent_model = mocked_pepinvent_model()
        self._model = PepinventAdapter(pepinvent_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
