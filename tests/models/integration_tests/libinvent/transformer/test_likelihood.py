import unittest

import pytest
import torch

from reinvent.models import LibinventTransformerAdapter, SampledSequencesDTO
from reinvent.models.transformer.libinvent.libinvent import LibinventModel
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE
from tests.test_data import SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, \
    SCAFFOLD_QUADRUPLE_POINT, DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestLibInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(SCAFFOLD_SINGLE_POINT, DECORATION_NO_SUZUKI, 0.4)
        dto2 = SampledSequencesDTO(SCAFFOLD_DOUBLE_POINT, TWO_DECORATIONS_ONE_SUZUKI, 0.6)
        dto3 = SampledSequencesDTO(SCAFFOLD_TRIPLE_POINT, THREE_DECORATIONS, 0.3)
        dto4 = SampledSequencesDTO(SCAFFOLD_QUADRUPLE_POINT, FOUR_DECORATIONS, 0.5)
        self.sampled_sequence_list = [dto1, dto2, dto3, dto4]
        save_dict = torch.load(self.json_config["LIBINVENT_PRIOR_PATH"], map_location=self.device, weights_only=False)
        model = LibinventModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = LibinventTransformerAdapter(model)

    def test_len_likelihood_smiles(self):
        results = self.adapter.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([4], list(results.likelihood.shape))
