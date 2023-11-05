import unittest

import numpy as np
import numpy.testing as npt
import torch.utils.data as tud

from reinvent.models import Mol2MolAdapter
from reinvent.models.mol2mol.dataset.dataset import Dataset
from reinvent.models.mol2mol.enums import SamplingModesEnum
from reinvent.models.mol2mol.models.vocabulary import SMILESTokenizer
from tests.test_data import BENZENE, TOLUENE, ANILINE
from tests.models.unit_tests.molformer.fixtures import mocked_molformer_model


class TestSampleLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        molformer_model = mocked_molformer_model()
        self._model = Mol2MolAdapter(molformer_model)

        self._sample_mode_enum = SamplingModesEnum()
        smiles_list = [BENZENE, TOLUENE, ANILINE]
        self.data_loader = self.initialize_dataloader(smiles_list)
        self.beamsize = 1
        self._model.set_beam_size(self.beamsize)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample_molecules(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.BEAMSEARCH)

    def test_sample_likelihood_smiles_consistency(self):
        sampled_sequence_list = self._sample_molecules(self.data_loader)
        mask = []
        for smi in sampled_sequence_list.items2:
            if len(set(smi)) == 1 and smi[0] == "*":
                mask.append(False)
            else:
                mask.append(True)

        sampled_nlls_list = []
        for sampled_sequence_dto in sampled_sequence_list:
            sampled_nlls_list.append(sampled_sequence_dto.nll)
        sampled_nlls_array = np.array(sampled_nlls_list)

        batch_likelihood_dto = self._model.likelihood_smiles(sampled_sequence_list)
        likelihood_smiles_nlls_array = batch_likelihood_dto.likelihood.cpu().detach().numpy()

        npt.assert_array_almost_equal(sampled_nlls_array[mask], likelihood_smiles_nlls_array[mask], decimal=2)
