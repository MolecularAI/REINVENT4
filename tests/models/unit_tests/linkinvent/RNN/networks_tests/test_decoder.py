import unittest

import torch.utils.data as tud
from torch import Tensor

from reinvent.models.linkinvent.dataset.paired_dataset import (
    PairedDataset,
    PairedModelVocabulary,
)
from reinvent.models.linkinvent.networks.decoder import Decoder
from reinvent.models.linkinvent.networks.encoder import Encoder
from tests.test_data import (
    ETHANE,
    METAMIZOLE,
    PROPANE,
    IBUPROFEN,
    BUTANE,
    ASPIRIN,
    WARHEAD_PAIR,
    CELECOXIB,
)


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles_list_input = [ETHANE, PROPANE, BUTANE, METAMIZOLE, IBUPROFEN]
        self.smiles_list_target = [ASPIRIN, WARHEAD_PAIR, CELECOXIB]

        paired_model_vocabulary = PairedModelVocabulary.from_lists(
            self.smiles_list_input, self.smiles_list_target
        )
        self.voc_size_input, self.voc_size_target = paired_model_vocabulary.len()
        dataset = PairedDataset(
            [[i, t] for i, t in zip(self.smiles_list_input, self.smiles_list_target)],
            vocabulary=paired_model_vocabulary,
        )

        self.data_loader_a = tud.DataLoader(
            dataset, batch_size=100, shuffle=False, collate_fn=dataset.collate_fn
        )
        self.data_loader_b = tud.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn
        )

        self.parameter_1 = dict(num_layers=3, num_dimensions=512, dropout=0)
        self.parameter_2 = dict(num_layers=1, num_dimensions=1, dropout=0)
        self.encoder_1 = Encoder(**self.parameter_1, vocabulary_size=self.voc_size_input)
        self.encoder_2 = Encoder(**self.parameter_2, vocabulary_size=self.voc_size_input)
        self.decoder_1 = Decoder(**self.parameter_1, vocabulary_size=self.voc_size_target)
        self.decoder_2 = Decoder(**self.parameter_2, vocabulary_size=self.voc_size_target)

    def test_get_params(self):
        p_dict = self.parameter_1
        p_dict["vocabulary_size"] = self.voc_size_target
        self.assertEqual(p_dict, self.decoder_1.get_params())

    def test_forward_normal_batch_no_singleton_dimensions(self):
        self._test_combination(self.data_loader_a, self.encoder_1, self.decoder_1, self.parameter_1)

    def test_forward_normal_batch_size_1_no_singleton_dimensions(self):
        self._test_combination(self.data_loader_b, self.encoder_1, self.decoder_1, self.parameter_1)

    def test_forward_normal_batch_with_singleton_dimensions(self):
        self._test_combination(self.data_loader_a, self.encoder_2, self.decoder_2, self.parameter_2)

    def test_forward_normal_batch_size_1_with_singleton_dimensions(self):
        self._test_combination(self.data_loader_b, self.encoder_2, self.decoder_2, self.parameter_2)

    def _test_combination(
        self,
        data_loader: tud.DataLoader,
        encoder: Encoder,
        decoder: Decoder,
        parameter: dict,
    ):

        for (seqs_input, seq_lengths_input), (
            seqs_target,
            seq_lengths_target,
        ) in data_loader:
            seqs_padded_enc, hidden_states = encoder.forward(seqs_input, seq_lengths_input)
            logits, hidden_states, weights = decoder.forward(
                seqs_target, seq_lengths_target, seqs_padded_enc, hidden_states
            )
            batch_size = min(data_loader.batch_size, len(self.smiles_list_target))
            expected_size_logits = [
                batch_size,
                seq_lengths_target.max().item(),
                self.voc_size_target,
            ]
            expected_size_hidden_state = [
                parameter["num_layers"],
                batch_size,
                parameter["num_dimensions"],
            ]
            expected_size_weights = [
                batch_size,
                seq_lengths_target.max().item(),
                seq_lengths_input.max().item(),
            ]

            self.assertTrue(all([isinstance(t, Tensor) for t in [logits, *hidden_states, weights]]))
            self.assertEqual(expected_size_logits, list(logits.size()))
            [self.assertEqual(expected_size_hidden_state, list(hs.size())) for hs in hidden_states]
            self.assertEqual(expected_size_weights, list(weights.size()))

            break  # enough to test first batch
