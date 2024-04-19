import pytest
import unittest

from reinvent.models import LinkinventTransformerAdapter, SampledSequencesDTO
from tests.models.unit_tests.linkinvent.transformer.fixtures import mocked_linkinvent_model
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE


@pytest.mark.usefixtures("device")
class TestLinkInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        linkinvent_model = mocked_linkinvent_model()
        self._model = LinkinventTransformerAdapter(linkinvent_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
