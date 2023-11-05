import unittest

import torch.utils.data as tud

from reinvent.models.reinvent.models.dataset import Dataset
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer
from tests.test_data import (
    SIMPLE_TOKENS,
    HEXANE,
    PROPANE,
    ETHANE,
    BUTANE,
    ASPIRIN,
)


class TestDatasetFunctions(unittest.TestCase):
    def setUp(self):
        self.smiles = [ASPIRIN, BUTANE, ETHANE, PROPANE, HEXANE]
        self.Dataset = Dataset(self.smiles, SIMPLE_TOKENS, SMILESTokenizer)
        self.coldata = tud.DataLoader(self.smiles, 1, collate_fn=Dataset.collate_fn)

    def test_dataset_functions_1(self):
        self.assertEqual(len(self.coldata), 5)
