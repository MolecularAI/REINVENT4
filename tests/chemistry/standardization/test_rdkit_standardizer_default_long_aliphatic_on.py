from reinvent.chemistry.standardization.filter_types_enum import FilterTypesEnum
from tests.chemistry.fixtures.test_data import INVALID
from tests.chemistry.standardization.base_rdkit_standardizer import BaseRDKitStandardizer


class TestRDKitStandardizerDefaultLongAliphaticOn(BaseRDKitStandardizer):
    def setUp(self):
        filter_types = FilterTypesEnum()
        self.raw_config = {
            "name": filter_types.DEFAULT,
            "parameters": {"remove_long_side_chains": True},
        }
        super().setUp()

    def test_standardizer_1(self):
        result = self.standardizer.apply_filter(INVALID)
        self.assertIsNone(result)
