import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from reinvent.models.linkinvent.model_vocabulary import ModelVocabulary
from reinvent.models.linkinvent.model_vocabulary.vocabulary import (
    SMILESTokenizer,
    create_vocabulary,
)
from tests.test_data import (
    CELECOXIB,
    COCAINE,
    METAMIZOLE,
    ASPIRIN,
    SCAFFOLD_SUZUKI,
    WARHEAD_PAIR,
    ETHANE,
)


class TestModelVocabulary(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles_list = [
            ASPIRIN,
            METAMIZOLE,
            COCAINE,
            CELECOXIB,
            WARHEAD_PAIR,
            SCAFFOLD_SUZUKI,
        ]
        self.model_voc = ModelVocabulary(
            create_vocabulary(self.smiles_list, SMILESTokenizer()), SMILESTokenizer()
        )

    def test_from_list(self):
        model_vocabulary = ModelVocabulary.from_list(self.smiles_list)
        self.assertTrue(isinstance(model_vocabulary, ModelVocabulary))
        self.assertEqual(model_vocabulary.vocabulary.tokens(), self.model_voc.vocabulary.tokens())

    def test_encode_decode(self):
        self.assertEqual(self.model_voc.decode(self.model_voc.encode(ASPIRIN)), ASPIRIN)
        self.assertEqual(self.model_voc.decode(self.model_voc.encode(WARHEAD_PAIR)), WARHEAD_PAIR)
        self.assertEqual(
            self.model_voc.decode(self.model_voc.encode(SCAFFOLD_SUZUKI)),
            SCAFFOLD_SUZUKI,
        )

    def test_encode(self):
        assert_almost_equal(self.model_voc.encode(ETHANE), [2, 10, 10, 1])
        self.assertIsNotNone(self.model_voc.encode("*[*]|"))

    def _decode_with_and_without_padding(self, smiles, encoded):
        self.assertEqual(self.model_voc.decode(encoded), smiles)
        self.assertEqual(self.model_voc.decode(np.append(encoded, [0, 0])), smiles)
        self.assertNotEqual(self.model_voc.decode(np.append([0, 0], encoded)), smiles)

    def test_decode(self):
        self._decode_with_and_without_padding(ETHANE, [2, 10, 10, 1])
        self._decode_with_and_without_padding(WARHEAD_PAIR, self.model_voc.encode(WARHEAD_PAIR))
        self._decode_with_and_without_padding(
            SCAFFOLD_SUZUKI, self.model_voc.encode(SCAFFOLD_SUZUKI)
        )

        self.assertEqual(self.model_voc.decode([0]), "<pad>")

    def test_len(self):
        self.assertEqual(len(self.model_voc), 19)
