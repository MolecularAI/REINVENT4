import unittest

from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer
from tests.chemistry.fixtures.test_data import CELECOXIB2


class TestRDKitStandardizerNoConfig(unittest.TestCase):
    def setUp(self):
        filter_configs = []
        self.standardizer = RDKitStandardizer(filter_configs)

        self.compound_1 = CELECOXIB2

    def test_standardizer_1(self):
        result = self.standardizer.apply_filter(self.compound_1)
        self.assertEqual(self.compound_1, result)
