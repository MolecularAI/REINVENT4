import unittest

from reinvent.chemistry import conversions
from reinvent.chemistry.library_design import FragmentFilter
from reinvent.chemistry.library_design.dtos import FilteringConditionDTO
from reinvent.chemistry.library_design.molecular_descriptors_enum import MolecularDescriptorsEnum
from tests.chemistry.fixtures.test_data import CELECOXIB, BENZENE, COCAINE_FRAGMENT


class TestFragmentFilter(unittest.TestCase):
    def setUp(self):
        descriptors_enum = MolecularDescriptorsEnum()
        condition_1 = FilteringConditionDTO(descriptors_enum.RING_COUNT, equals=3)
        condition_2 = FilteringConditionDTO(descriptors_enum.CLOGP, min=0.9)
        conditions = [condition_1, condition_2]
        self.filter = FragmentFilter(conditions)

    def test_compliant_with_conditions(self):
        smile = COCAINE_FRAGMENT
        molecule = conversions.smile_to_mol(smile)

        result = self.filter.filter(molecule)
        self.assertTrue(result)

    def test_non_compliant_with_conditions(self):
        smile = CELECOXIB
        molecule = conversions.smile_to_mol(smile)

        result = self.filter.filter(molecule)
        self.assertFalse(result)

    def test_non_compliant_with_ring_count(self):
        smile = BENZENE
        molecule = conversions.smile_to_mol(smile)

        result = self.filter.filter(molecule)
        self.assertFalse(result)
