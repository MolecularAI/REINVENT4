import pytest
import unittest

import numpy as np
import numpy.testing as npt
import torch
import torch.utils.data as tud

from reinvent.models import Mol2MolAdapter
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.utils import set_torch_device
from tests.models.unit_tests.mol2mol.fixtures import mocked_mol2mol_model
from tests.test_data import BENZENE, TOLUENE, ANILINE


@pytest.mark.usefixtures("device")
class TestSampleLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        device = torch.device(self.device)
        mol2mol_model = mocked_mol2mol_model()
        mol2mol_model.network.to(device)
        mol2mol_model.device = device

        set_torch_device(device)
        self._model = Mol2MolAdapter(mol2mol_model)

        self._sample_mode_enum = SamplingModesEnum()
        smiles_list = [BENZENE, TOLUENE, ANILINE]
        self.data_loader = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )

        return dataloader

    def _sample_molecules(self, data_loader):
        for batch in data_loader:
            return self._model.sample(*batch, decode_type=self._sample_mode_enum.MULTINOMIAL)

    def _sample_nlls(self, sampled_sequence_list):
        sampled_nlls_list = []
        for sampled_sequence_dto in sampled_sequence_list:
            sampled_nlls_list.append(sampled_sequence_dto.nll)
        sampled_nlls_array = np.array(sampled_nlls_list)
        return sampled_nlls_array

    def test_sample_likelihood_smiles_consistency(self):
        sampled_sequence_list = self._sample_molecules(self.data_loader)
        sampled_nlls_array = self._sample_nlls(sampled_sequence_list)

        batch_likelihood_dto = self._model.likelihood_smiles(sampled_sequence_list)
        likelihood_smiles_nlls_array = batch_likelihood_dto.likelihood.cpu().detach().numpy()

        npt.assert_array_almost_equal(sampled_nlls_array, likelihood_smiles_nlls_array, decimal=4)

    def test_sample_likelihood_smiles_consistency_temperature(self):
        self._model.set_temperature(1.5)

        sampled_sequence_list = self._sample_molecules(self.data_loader)
        sampled_nlls_array = self._sample_nlls(sampled_sequence_list)

        batch_likelihood_dto = self._model.likelihood_smiles(sampled_sequence_list)
        likelihood_smiles_nlls_array = batch_likelihood_dto.likelihood.cpu().detach().numpy()

        npt.assert_array_almost_equal(sampled_nlls_array, likelihood_smiles_nlls_array, decimal=4)
