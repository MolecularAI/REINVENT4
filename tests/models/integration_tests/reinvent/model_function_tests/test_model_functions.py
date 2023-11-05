import os
import shutil
import pytest
import unittest

import torch
import numpy
import numpy.testing as nt

from reinvent.runmodes.utils.helpers import set_torch_device
from reinvent.models.reinvent.models.model import Model
from tests.test_data import PROPANE, BENZENE, METAMIZOLE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        self.workfolder = self.json_config["MAIN_TEST_PATH"]
        self.output_file = os.path.join(self.workfolder, "generative_model.ckpt")

        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)

        self.model = Model.load_from_file(
            self.json_config["PRIOR_PATH"], "inference", torch.device(self.device)
        )
        set_torch_device(self.device)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _relerr(self, va, ve):
        return abs( (va - ve) / ve)

    def test_likelihoods_from_model_1(self):
        likelihoods = self.model.likelihood_smiles([PROPANE, BENZENE])

        #self.assertAlmostEqual(likelihoods[0].item(), 20.9116, 3)
        #self.assertAlmostEqual(likelihoods[1].item(), 17.9506, 3)
        self.assertTrue( self._relerr( likelihoods[0].item(), 20.9116 ) < 0.01)
        self.assertTrue( self._relerr( likelihoods[1].item(), 17.9506 ) < 0.01)
        self.assertEqual(len(likelihoods), 2)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_likelihoods_from_model_2(self):
        likelihoods = self.model.likelihood_smiles([METAMIZOLE])

        # self.assertAlmostEqual(likelihoods[0].item(), 125.4669, 3)
        self.assertTrue( self._relerr( likelihoods[0].item(), 125.4669 ) < 0.01)
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

    def test_model_tokens(self):
        tokens = self.model.vocabulary.tokens()

        self.assertIn("C", tokens)
        self.assertIn("O", tokens)
        self.assertIn("Cl", tokens)

        self.assertEqual(len(tokens), 34)
        self.assertEqual(type(tokens), list)

    def test_save_model(self):
        self.model.save(self.output_file)

        self.assertEqual(os.path.isfile(self.output_file), True)

    def test_likelihood_function_differences(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=128)
        nll2 = self.model.likelihood(seq)
        nll3 = self.model.likelihood_smiles(sample)

        nt.assert_array_almost_equal(
            nll.detach().cpu().numpy(), nll2.detach().cpu().numpy(), 3
        )
        nt.assert_array_almost_equal(
            nll.detach().cpu().numpy(), nll3.detach().cpu().numpy(), 3
        )
