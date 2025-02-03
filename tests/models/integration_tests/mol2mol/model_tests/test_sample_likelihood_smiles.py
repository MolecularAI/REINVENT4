import unittest

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.utils.data as tud

from reinvent.models import TransformerAdapter, Mol2MolModel
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import METAMIZOLE, COCAINE, AMOXAPINE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestSampleLikelihoodSMILES(unittest.TestCase):
    def setUp(self):

        save_dict = torch.load(self.json_config["MOLFORMER_PRIOR_PATH"], weights_only=False)
        model = Mol2MolModel.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)

        self.adapter = TransformerAdapter(model)

        smiles_list = [METAMIZOLE, COCAINE, AMOXAPINE]
        self.data_loader = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self.adapter.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn
        )
        return dataloader

    def _sample_molecules(self, data_loader):
        results = []
        for batch in data_loader:
            src, src_mask = batch
            for sampled_sequence in self.adapter.sample(
                src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL
            ):
                results.append(sampled_sequence)
        return results

    def test_sample_likelihood_smiles_consistency(self):
        sampled_sequence_list = self._sample_molecules(self.data_loader)

        sampled_nlls_list = []
        for sampled_sequence_dto in sampled_sequence_list:
            sampled_nlls_list.append(sampled_sequence_dto.nll)
        sampled_nlls_array = np.array(sampled_nlls_list)

        batch_likelihood_dto = self.adapter.likelihood_smiles(sampled_sequence_list)
        likelihood_smiles_nlls_array = batch_likelihood_dto.likelihood.cpu().detach().numpy()

        npt.assert_array_almost_equal(sampled_nlls_array, likelihood_smiles_nlls_array, decimal=4)
