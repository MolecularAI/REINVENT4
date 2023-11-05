import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from reinvent.chemistry import Conversions, PhysChemDescriptors
from tests.chemistry.fixtures.test_data import (
    ASPIRIN,
    CELECOXIB,
    IBUPROFEN,
    PENTANE,
    METAMIZOLE,
    METHYLPHEMYL_FRAGMENT,
    GENTAMICIN,
)


class TestPhysChemDescriptors(unittest.TestCase):
    def setUp(self) -> None:
        self.conversions = Conversions()
        self.descriptor = PhysChemDescriptors()

        self.smiles_list = [
            ASPIRIN,
            CELECOXIB,
            IBUPROFEN,
            PENTANE,
            METAMIZOLE,
            METHYLPHEMYL_FRAGMENT,
            GENTAMICIN,
        ]
        self.expected_values = dict(
            number_atoms_in_largest_ring=[6, 6, 6, 0, 6, 6, 6],
            maximum_graph_length=[6, 12, 9, 4, 10, 5, 15],
            hba_libinski=[3, 4, 1, 0, 6, 0, 12],
            hbd_libinski=[1, 1, 1, 0, 1, 0, 8],
            mol_weight=[180.159, 381.379, 206.285, 72.151, 311.363, 91.133, 477.603],
            number_of_rings=[1, 3, 1, 0, 2, 1, 3],
            number_of_rotatable_bonds=[2, 3, 4, 2, 4, 0, 7],
            slog_p=[1.31, 3.514, 3.073, 2.197, 0.766, 1.17, -3.327],
            tpsa=[63.6, 77.98, 37.3, 0.0, 84.54, 0.0, 199.73],
            number_of_stereo_centers=[0, 0, 1, 0, 0, 0, 13],
        )
        self.mol_list = [self.conversions.smile_to_mol(smiles) for smiles in self.smiles_list]

    def _test_property(self, property_name):
        values = np.array([getattr(self.descriptor, property_name)(mol) for mol in self.mol_list])
        assert_almost_equal(values, self.expected_values.get(property_name), 3)

    def test_maximum_graph_length(self):
        self._test_property("maximum_graph_length")

    def test_number_atoms_in_largest_ring(self):
        self._test_property("number_atoms_in_largest_ring")

    def test_hba_libinski(self):
        self._test_property("hba_libinski")

    def test_hbd_libinski(self):
        self._test_property("hbd_libinski")

    def test_mol_weight(self):
        self._test_property("mol_weight")

    def test_number_of_rings(self):
        self._test_property("number_of_rings")

    def test_number_of_rotatable_bonds(self):
        self._test_property("number_of_rotatable_bonds")

    def test_slog_p(self):
        self._test_property("slog_p")

    def test_tpsa(self):
        self._test_property("tpsa")

    def test_number_of_stereo_centers(self):
        self._test_property("number_of_stereo_centers")
