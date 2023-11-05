import unittest

from reinvent.chemistry import Conversions
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer
from tests.chemistry.fixtures.test_data import CELECOXIB2
from tests.chemistry.standardization.fixtures import MockLogger


class TestRDKitStandardizerNoConfig(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        logger = MockLogger()
        filter_configs = []
        self.standardizer = RDKitStandardizer(filter_configs, logger)

        self.compound_1 = CELECOXIB2

    def test_standardizer_1(self):
        result = self.standardizer.apply_filter(self.compound_1)
        self.assertEqual(self.compound_1, result)
