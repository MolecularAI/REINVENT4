import unittest

import reinvent.models.linkinvent.model_vocabulary.vocabulary as mv
from tests.test_data import CELECOXIB, COCAINE, METAMIZOLE, ASPIRIN, SIMPLE_TOKENS


class TestCreateVocabulary(unittest.TestCase):
    def setUp(self):
        smiles = [ASPIRIN, METAMIZOLE, COCAINE, CELECOXIB]
        self.voc = mv.create_vocabulary(smiles_list=smiles, tokenizer=mv.SMILESTokenizer())

    def test_create(self):
        simple_vocabulary = mv.Vocabulary(tokens=SIMPLE_TOKENS)
        self.assertEqual(self.voc.tokens(), ["<pad>"] + simple_vocabulary.tokens())
