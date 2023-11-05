import unittest

from tests.chemistry.fixtures.test_data import REP_SMILES_LIST, INVALID_SMILES_LIST
from reinvent.chemistry.utils import get_indices_of_unique_smiles


class TestUtilsUniqueSeqs(unittest.TestCase):
    def test_duplicate_seqs_list(self):
        seqs = REP_SMILES_LIST
        idxs = get_indices_of_unique_smiles(seqs)
        self.assertEqual(len(idxs), 2)
        self.assertEqual(seqs[idxs[0]], seqs[0])
        self.assertEqual(seqs[idxs[1]], seqs[15])

    def test_invalid_seqs_list(self):
        seqs = INVALID_SMILES_LIST
        idxs = get_indices_of_unique_smiles(seqs)
        self.assertEqual(len(idxs), 1)
        self.assertEqual(seqs[idxs[0]], seqs[0])
