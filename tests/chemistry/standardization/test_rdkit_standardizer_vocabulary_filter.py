from reinvent.chemistry.standardization.filter_types_enum import FilterTypesEnum
from tests.chemistry.fixtures.test_data import (
    IBUPROFEN,
    METHYL_3_O_TOLYL_PROPYL_AMINE,
    METHYL_3_O_TOLYL_PROPYL_AMINE2,
)
from tests.chemistry.standardization.base_rdkit_standardizer import BaseRDKitStandardizer


class TestRDKitStandardizerVocabularyFilter(BaseRDKitStandardizer):
    def setUp(self):
        filter_types = FilterTypesEnum()
        self.raw_config = {
            "name": filter_types.VOCABULARY_FILTER,
            "parameters": {"vocabulary": ["C", "c", "1", "N", "[CH]"]},
        }
        super().setUp()

    def test_standardizer_positive(self):
        result = self.standardizer.apply_filter(METHYL_3_O_TOLYL_PROPYL_AMINE2)
        self.assertEqual(METHYL_3_O_TOLYL_PROPYL_AMINE, result)

    def test_standardizer_negative(self):
        result = self.standardizer.apply_filter(IBUPROFEN)
        self.assertIsNone(result)
