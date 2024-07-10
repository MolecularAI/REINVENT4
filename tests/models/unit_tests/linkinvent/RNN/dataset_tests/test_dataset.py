import unittest

from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.linkinvent.model_vocabulary.model_vocabulary import ModelVocabulary
from tests.test_data import ETHANE, HEXANE, PROPANE, BUTANE


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles = [ETHANE, HEXANE, PROPANE, BUTANE]
        self.model_voc = ModelVocabulary.from_list(self.smiles)
        self.data_set = Dataset(self.smiles, self.model_voc)
        self.padded_seq, self.seq_length = Dataset.collate_fn(self.data_set)

    def test_len(self):
        self.assertEqual(len(self.data_set), 4)

    def test_coll_fn(self):
        self.assertEqual(len(self.padded_seq), len(self.smiles))
        self.assertEqual(len(self.padded_seq[0]), len(self.padded_seq[1]))
