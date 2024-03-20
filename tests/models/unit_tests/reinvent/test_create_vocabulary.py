import unittest

from reinvent.models.reinvent.models.vocabulary import (
    Vocabulary,
    create_vocabulary,
    SMILESTokenizer,
)
from tests.test_data import CELECOXIB, COCAINE, METAMIZOLE, ASPIRIN, SIMPLE_TOKENS


class TestCreateVocabulary(unittest.TestCase):
    def setUp(self):
        smiles = [ASPIRIN, METAMIZOLE, COCAINE, CELECOXIB]
        self.voc = create_vocabulary(smiles_list=smiles, tokenizer=SMILESTokenizer())

    def test_create(self):
        simple_vocabulary = Vocabulary(tokens=SIMPLE_TOKENS)
        self.assertEqual(self.voc.tokens(), simple_vocabulary.tokens())
        self.assertEqual(self.voc, simple_vocabulary)
