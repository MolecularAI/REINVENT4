import pytest
import unittest

from reinvent.models import LibinventTransformerAdapter, SampledSequencesDTO
from tests.models.unit_tests.libinvent.transformer.fixtures import mocked_libinvent_model
from tests.test_data import SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, \
    SCAFFOLD_QUADRUPLE_POINT, DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS


@pytest.mark.usefixtures("device")
class TestLibInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(SCAFFOLD_SINGLE_POINT, DECORATION_NO_SUZUKI, 0.4)
        dto2 = SampledSequencesDTO(SCAFFOLD_DOUBLE_POINT, TWO_DECORATIONS_ONE_SUZUKI, 0.6)
        dto3 = SampledSequencesDTO(SCAFFOLD_TRIPLE_POINT, THREE_DECORATIONS, 0.3)
        dto4 = SampledSequencesDTO(SCAFFOLD_QUADRUPLE_POINT, FOUR_DECORATIONS, 0.5)
        self.sampled_sequence_list = [dto1, dto2, dto3, dto4]

        libinvent_model = mocked_libinvent_model()
        self._model = LibinventTransformerAdapter(libinvent_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([4], list(results.likelihood.shape))
