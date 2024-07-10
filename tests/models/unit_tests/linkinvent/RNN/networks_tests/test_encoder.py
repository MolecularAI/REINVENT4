import unittest

import torch.utils.data as tud
from torch import Tensor

from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.linkinvent.model_vocabulary import ModelVocabulary
from reinvent.models.linkinvent.model_vocabulary.vocabulary import (
    SMILESTokenizer,
    create_vocabulary,
)
from reinvent.models.linkinvent.networks.encoder import Encoder
from tests.test_data import ETHANE, PROPANE, BUTANE


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        smiles_list = [ETHANE, PROPANE, BUTANE]
        model_vocabulary = ModelVocabulary(
            create_vocabulary(smiles_list, SMILESTokenizer()), SMILESTokenizer()
        )
        vocabulary_size = len(model_vocabulary)
        dataset = Dataset(smiles_list, model_vocabulary)

        self.data_loader_full_smiles_list = tud.DataLoader(
            dataset,
            batch_size=len(smiles_list),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        self.data_loader_batch_size_1 = tud.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn
        )

        self.parameter_1 = dict(
            num_layers=3, num_dimensions=512, vocabulary_size=vocabulary_size, dropout=0
        )
        self.parameter_2 = dict(
            num_layers=1, num_dimensions=1, vocabulary_size=vocabulary_size, dropout=0
        )
        self.encoder_1 = Encoder(**self.parameter_1)
        self.encoder_2 = Encoder(**self.parameter_2)

    def test_get_params(self):
        self.assertEqual(self.parameter_1, self.encoder_1.get_params())
        self.assertEqual(self.parameter_2, self.encoder_2.get_params())

    def test_forward_normal_batch_size(self):
        self._test_output_(
            self.encoder_1,
            self.data_loader_full_smiles_list,
            self.parameter_1["num_dimensions"],
            self.parameter_1["num_layers"],
        )

    def test_forward_normal_batch_size_singleton_dimensions(self):
        self._test_output_(
            self.encoder_2,
            self.data_loader_full_smiles_list,
            self.parameter_2["num_dimensions"],
            self.parameter_2["num_layers"],
        )

    def test_forward_batch_size_of_1(self):
        self._test_output_(
            self.encoder_1,
            self.data_loader_batch_size_1,
            self.parameter_1["num_dimensions"],
            self.parameter_1["num_layers"],
        )

    def test_forward_batch_size_of_1_singleton_dimensions(self):
        self._test_output_(
            self.encoder_2,
            self.data_loader_batch_size_1,
            self.parameter_2["num_dimensions"],
            self.parameter_2["num_layers"],
        )

    def _test_output_(
        self,
        encoder: Encoder,
        data_loader: tud.DataLoader,
        num_dimensions: int,
        num_layers: int,
    ):
        for padded_seqs, seqs_length in data_loader:

            encoder_padded_seqs, (hidden_1, hidden_2) = encoder.forward(padded_seqs, seqs_length)
            expected_size_encoder_padded_seqs = [
                data_loader.batch_size,
                seqs_length.max().item(),
                num_dimensions,
            ]
            expected_size_hidden = [num_layers, data_loader.batch_size, num_dimensions]

            # check that all output items are Tensors
            self.assertTrue(
                all(
                    [isinstance(item, Tensor) for item in [encoder_padded_seqs, hidden_1, hidden_2]]
                )
            )

            # check the size of each Tensor
            self.assertEqual(expected_size_encoder_padded_seqs, list(encoder_padded_seqs.size()))
            self.assertEqual(expected_size_hidden, list(hidden_1.size()))
            self.assertEqual(expected_size_hidden, list(hidden_2.size()))

            break  # only check first batch
