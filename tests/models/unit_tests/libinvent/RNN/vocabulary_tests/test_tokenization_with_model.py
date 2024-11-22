import unittest

from tests.models.unit_tests.libinvent.RNN.fixtures import mocked_decorator_model


class TestTokenizationWithModel(unittest.TestCase):
    def setUp(self):
        self.smiles = "c1ccccc1CC0C"
        self.actor = mocked_decorator_model()

    def test_tokenization(self):
        tokenized = self.actor.vocabulary.scaffold_tokenizer.tokenize(self.smiles)
        self.assertEqual(14, len(tokenized))
