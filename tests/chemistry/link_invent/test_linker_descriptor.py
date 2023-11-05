import unittest

from rdkit.Chem import MolFromSmiles

from reinvent.chemistry.link_invent.linker_descriptors import LinkerDescriptors
from tests.chemistry.fixtures.test_data import (
    SCAFFOLD_TO_DECORATE,
    METAMIZOLE_SCAFFOLD_LABELED,
    METAMIZOLE_LABELED_PARTS,
    LINKER_WITH_RINGS_MOL,
    LINEAR_LINKER_MOL_1,
    LINEAR_LINKER_MOL_2,
    LINKER_MIXED_HYBRIDIZATION_MOL,
    LINKER_HBD_HBA_MOL,
)


class TestLinkerDescriptors(unittest.TestCase):
    def setUp(self) -> None:
        self.linear_linker_mol_1 = MolFromSmiles(LINEAR_LINKER_MOL_1)
        self.linear_linker_mol_2 = MolFromSmiles(LINEAR_LINKER_MOL_2)
        # linker below has 1 aromatic ring and 1 aliphatic ring
        self.linker_with_rings_mol = MolFromSmiles(LINKER_WITH_RINGS_MOL)
        # linker below has 3 sp3 atom, 2 sp2 atoms, and 2 sp3 atoms
        self.linker_mixed_hybridization_mol = MolFromSmiles(LINKER_MIXED_HYBRIDIZATION_MOL)
        # linker below has 1 hydrogen bond donor and 1 hydrogen bond acceptor
        self.linker_hbd_hba_mol = MolFromSmiles(LINKER_HBD_HBA_MOL)
        # this specific molecular tests whether the linker capping with explicit H is performed correctly
        # if not, certain rdkit calculations will fail, e.g., num_rings
        self.metamizole_mol = MolFromSmiles(METAMIZOLE_LABELED_PARTS)

        self.scaffold_to_decorate = SCAFFOLD_TO_DECORATE
        self.metamizole_scaffold_labeled = METAMIZOLE_SCAFFOLD_LABELED
        self.linker_mixed_hybridization_smiles = LINKER_MIXED_HYBRIDIZATION_MOL
        self.linker_hba_hba_smiles = LINKER_HBD_HBA_MOL

        self.linker_descriptor = LinkerDescriptors()

    def test_effective_length(self):
        self.assertEqual(self.linker_descriptor.effective_length(self.linear_linker_mol_1), 5)
        self.assertEqual(self.linker_descriptor.effective_length(self.linear_linker_mol_2), 1)

    def test_max_graph_length(self):
        self.assertEqual(self.linker_descriptor.max_graph_length(self.linear_linker_mol_1), 5)
        self.assertEqual(self.linker_descriptor.max_graph_length(self.linear_linker_mol_2), 3)

    def test_length_ratio(self):
        self.assertEqual(self.linker_descriptor.length_ratio(self.linear_linker_mol_1), 100)
        self.assertAlmostEqual(
            self.linker_descriptor.length_ratio(self.linear_linker_mol_2), 33.333, 3
        )

    def test_num_rings(self):
        self.assertEqual(self.linker_descriptor.num_rings(self.linear_linker_mol_1), 0)
        self.assertEqual(self.linker_descriptor.num_rings(self.linker_with_rings_mol), 2)

        self.assertEqual(self.linker_descriptor.num_rings(self.metamizole_mol), 1)

    def test_num_aromatic_rings(self):
        self.assertEqual(self.linker_descriptor.num_aromatic_rings(self.linear_linker_mol_1), 0)
        self.assertEqual(self.linker_descriptor.num_aromatic_rings(self.linker_with_rings_mol), 1)

        self.assertEqual(self.linker_descriptor.num_aromatic_rings(self.metamizole_mol), 1)

    def test_num_aliphatic_rings(self):
        self.assertEqual(self.linker_descriptor.num_aliphatic_rings(self.linear_linker_mol_1), 0)
        self.assertEqual(self.linker_descriptor.num_aliphatic_rings(self.linker_with_rings_mol), 1)

        self.assertEqual(self.linker_descriptor.num_aliphatic_rings(self.metamizole_mol), 0)

    def test_num_sp_atoms(self):
        self.assertEqual(self.linker_descriptor.num_sp_atoms(self.linear_linker_mol_1), 0)
        self.assertEqual(
            self.linker_descriptor.num_sp_atoms(self.linker_mixed_hybridization_mol), 2
        )

    def test_num_sp2_atoms(self):
        self.assertEqual(self.linker_descriptor.num_sp2_atoms(self.linear_linker_mol_1), 0)
        self.assertEqual(
            self.linker_descriptor.num_sp2_atoms(self.linker_mixed_hybridization_mol), 2
        )

    def test_num_sp3_atoms(self):
        self.assertEqual(self.linker_descriptor.num_sp3_atoms(self.linear_linker_mol_1), 6)
        self.assertEqual(
            self.linker_descriptor.num_sp3_atoms(self.linker_mixed_hybridization_mol), 3
        )

    def test_num_hbd(self):
        self.assertEqual(self.linker_descriptor.num_hbd(self.linear_linker_mol_1), 0)
        self.assertEqual(self.linker_descriptor.num_hbd(self.linker_hbd_hba_mol), 1)

    def test_num_hba(self):
        self.assertEqual(self.linker_descriptor.num_hba(self.linear_linker_mol_1), 0)
        self.assertEqual(self.linker_descriptor.num_hba(self.linker_hbd_hba_mol), 1)

    def test_mol_weight(self):
        self.assertAlmostEqual(
            self.linker_descriptor.mol_weight(self.linear_linker_mol_1), 86.178, 3
        )
        self.assertAlmostEqual(
            self.linker_descriptor.mol_weight(self.linker_hbd_hba_mol), 101.149, 3
        )

    def test_ratio_rotatable_bonds(self):
        self.assertAlmostEqual(
            self.linker_descriptor.ratio_rotatable_bonds(self.linker_mixed_hybridization_mol),
            16.667,
            3,
        )
        self.assertAlmostEqual(
            self.linker_descriptor.ratio_rotatable_bonds(self.linker_hbd_hba_mol), 33.333, 3
        )

    def test_effective_length_from_smile(self):
        self.assertEqual(
            self.linker_descriptor.effective_length_from_smile(self.scaffold_to_decorate), 6
        )
        self.assertEqual(
            self.linker_descriptor.effective_length_from_smile(self.metamizole_scaffold_labeled), 4
        )

    def test_max_graph_length_from_smile(self):
        self.assertEqual(
            self.linker_descriptor.max_graph_length_from_smile(self.scaffold_to_decorate), 11
        )
        self.assertEqual(
            self.linker_descriptor.max_graph_length_from_smile(self.metamizole_scaffold_labeled), 5
        )

    def test_length_ratio_from_smile(self):
        self.assertAlmostEqual(
            54.545, self.linker_descriptor.length_ratio_from_smiles(self.scaffold_to_decorate), 3
        )
        self.assertEqual(
            80, self.linker_descriptor.length_ratio_from_smiles(self.metamizole_scaffold_labeled)
        )

    def test_num_rings_from_smiles(self):
        self.assertEqual(self.linker_descriptor.num_rings_from_smiles(self.scaffold_to_decorate), 3)
        self.assertEqual(
            self.linker_descriptor.num_rings_from_smiles(self.metamizole_scaffold_labeled), 1
        )

    def test_num_aromatic_rings_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_aromatic_rings_from_smiles(self.scaffold_to_decorate), 3
        )
        self.assertEqual(
            self.linker_descriptor.num_aromatic_rings_from_smiles(self.metamizole_scaffold_labeled),
            1,
        )

    def test_num_aliphatic_rings_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_aliphatic_rings_from_smiles(self.scaffold_to_decorate), 0
        )
        self.assertEqual(
            self.linker_descriptor.num_aliphatic_rings_from_smiles(
                self.metamizole_scaffold_labeled
            ),
            0,
        )

    def test_num_sp_atoms_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_sp_atoms_from_smiles(self.scaffold_to_decorate), 0
        )
        self.assertEqual(
            self.linker_descriptor.num_sp_atoms_from_smiles(self.metamizole_scaffold_labeled), 0
        )

    def test_num_sp2_atoms_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_sp2_atoms_from_smiles(self.scaffold_to_decorate), 19
        )
        self.assertEqual(
            self.linker_descriptor.num_sp2_atoms_from_smiles(self.metamizole_scaffold_labeled), 7
        )

    def test_num_sp3_atoms_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_sp3_atoms_from_smiles(self.scaffold_to_decorate), 2
        )
        self.assertEqual(
            self.linker_descriptor.num_sp3_atoms_from_smiles(self.metamizole_scaffold_labeled), 4
        )

    def test_num_hbd_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_hbd_from_smiles(self.linker_mixed_hybridization_smiles), 0
        )
        self.assertEqual(self.linker_descriptor.num_hbd_from_smiles(self.linker_hba_hba_smiles), 1)

    def test_num_hba_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.num_hba_from_smiles(self.linker_mixed_hybridization_smiles), 0
        )
        self.assertEqual(self.linker_descriptor.num_hba_from_smiles(self.linker_hba_hba_smiles), 1)

    def test_mol_weight_from_smiles(self):
        self.assertAlmostEqual(
            self.linker_descriptor.mol_weight_from_smiles(self.scaffold_to_decorate), 297.339, 3
        )
        self.assertAlmostEqual(
            self.linker_descriptor.mol_weight_from_smiles(self.linker_hba_hba_smiles), 157.257
        )

    def test_ratio_rotatable_bonds_from_smiles(self):
        self.assertEqual(
            self.linker_descriptor.ratio_rotatable_bonds_from_smiles(
                self.linker_mixed_hybridization_smiles
            ),
            40.0,
        )
        self.assertEqual(
            self.linker_descriptor.ratio_rotatable_bonds_from_smiles(self.linker_hba_hba_smiles),
            60.0,
        )
