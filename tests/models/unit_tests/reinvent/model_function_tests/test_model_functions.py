import unittest

import numpy
import torch

from reinvent.models.reinvent.models.model import Model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer, Vocabulary
from tests.test_data import PROPANE, BENZENE, METAMIZOLE, SIMPLE_TOKENS


class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        vocabulary = Vocabulary(tokens=SIMPLE_TOKENS)
        tokenizer = SMILESTokenizer()
        self.model = Model(vocabulary, tokenizer)

    def test_likelihoods_from_model_1(self):
        likelihoods = self.model.likelihood_smiles([PROPANE, BENZENE])
        self.assertEqual(len(likelihoods), 2)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_likelihoods_from_model_2(self):
        likelihoods = self.model.likelihood_smiles([METAMIZOLE])
        self.assertEqual(len(likelihoods), 1)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_sample_from_model_1(self):
        sample, nll = self.model.sample_smiles(num=20, batch_size=20)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), numpy.ndarray)

    def test_sample_from_model_2(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=20)
        self.assertEqual(seq.shape[0], 20)
        self.assertEqual(type(seq), torch.Tensor)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), torch.Tensor)
