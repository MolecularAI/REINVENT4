from reinvent.chemistry.enums import FilterTypesEnum
from tests.chemistry.fixtures.test_data import CELECOXIB
from tests.chemistry.standardization.base_rdkit_standardizer import BaseRDKitStandardizer


class TestRDKitStandardizerDefaultLongAliphaticOff(BaseRDKitStandardizer):
    def setUp(self):
        filter_types = FilterTypesEnum()
        self.raw_config = {
            "name": filter_types.DEFAULT,
            "parameters": {"remove_long_side_chains": False},
        }
        super().setUp()

    def test_standardizer_1(self):
        result = self.standardizer.apply_filter(CELECOXIB)
        self.assertIsNotNone(result)
