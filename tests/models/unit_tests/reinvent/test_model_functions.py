import pytest
import unittest

import numpy
import torch

from reinvent.models import meta_data
from reinvent.models.reinvent.models.model import Model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer, Vocabulary
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import PROPANE, BENZENE, METAMIZOLE, SIMPLE_TOKENS


@pytest.mark.usefixtures("device")
class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        vocabulary = Vocabulary(tokens=SIMPLE_TOKENS)
        tokenizer = SMILESTokenizer()
        metadata = meta_data.ModelMetaData(
            hash_id=None, hash_id_format=0, model_id=0, origina_data_source="", creation_date=0
        )
        device = torch.device(self.device)
        self.model = Model(vocabulary, tokenizer, metadata, device=device)

        set_torch_device(device)

    def test_likelihoods_from_model_1(self):
        likelihoods = self.model.likelihood_smiles([PROPANE, BENZENE])
        self.assertEqual(len(likelihoods), 2)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_likelihoods_from_model_2(self):
        likelihoods = self.model.likelihood_smiles([METAMIZOLE])
        self.assertEqual(len(likelihoods), 1)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_sample_from_model(self):
        seq, sample, nll = self.model.sample(batch_size=20)
        self.assertEqual(seq.shape[0], 20)
        self.assertEqual(type(seq), torch.Tensor)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), torch.Tensor)
