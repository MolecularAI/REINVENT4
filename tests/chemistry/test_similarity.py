import unittest

from numpy import testing

from reinvent.chemistry import conversions
from reinvent.chemistry.similarity import calculate_tanimoto
from tests.chemistry.fixtures.test_data import ASPIRIN, CELECOXIB


class Test_similarity(unittest.TestCase):
    def setUp(self):
        self.aspirin_fp = conversions.smiles_to_fingerprints([ASPIRIN])
        self.celecoxib_fp = conversions.smiles_to_fingerprints([CELECOXIB])

    def test_calculate_tanimoto(self):
        score = calculate_tanimoto(self.celecoxib_fp, self.aspirin_fp)

        testing.assert_almost_equal(score, 0.1455, 3)
