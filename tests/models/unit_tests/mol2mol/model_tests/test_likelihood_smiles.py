import unittest

from reinvent.models import Mol2MolAdapter, SampledSequencesDTO
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE
from tests.models.unit_tests.mol2mol.fixtures import mocked_mol2mol_model


class TestLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]

        molformer_model = mocked_mol2mol_model()
        self._model = Mol2MolAdapter(molformer_model)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))
