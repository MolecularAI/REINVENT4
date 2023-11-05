import unittest

from tests.chemistry.fixtures.test_data import (
    ASPIRIN,
    CELECOXIB,
    ETHANE,
    IBUPROFEN,
    CAFFEINE,
    REP_SMILES_LIST,
    INVALID_SMILES_LIST,
)
from reinvent.chemistry.utils import get_indices_of_unique_smiles


class TestUtilsUniqueSmiles(unittest.TestCase):
    def test_unique_smiles_list(self):
        smiles = [ASPIRIN, CELECOXIB, ETHANE, IBUPROFEN, CAFFEINE]
        idxs = get_indices_of_unique_smiles(smiles)
        self.assertEqual(len(idxs), 5)
        self.assertEqual(smiles[idxs[0]], smiles[0])
        self.assertEqual(smiles[idxs[3]], smiles[3])

    def test_duplicate_smiles_list(self):
        smiles = REP_SMILES_LIST
        idxs = get_indices_of_unique_smiles(smiles)
        self.assertEqual(len(idxs), 2)
        self.assertEqual(smiles[idxs[0]], smiles[0])
        self.assertEqual(smiles[idxs[1]], smiles[15])

    def test_invalid_smiles_list(self):
        smiles = INVALID_SMILES_LIST
        idxs = get_indices_of_unique_smiles(smiles)
        self.assertEqual(len(idxs), 1)
        self.assertEqual(smiles[idxs[0]], smiles[0])
