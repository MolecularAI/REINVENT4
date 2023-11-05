import unittest

from rdkit.Chem import MolFromSmiles

from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import AttachmentPoints
from reinvent.chemistry.link_invent.bond_breaker import BondBreaker
from tests.chemistry.fixtures.test_data import METAMIZOLE_LABELED_COMPONENTS, METAMIZOLE_SCAFFOLD_FRAGMENT


class TestBondBreaker(unittest.TestCase):
    def setUp(self) -> None:
        self.conversions = Conversions()
        self.attachment_points = AttachmentPoints()
        self.bb = BondBreaker()
        self.mol = MolFromSmiles(METAMIZOLE_LABELED_COMPONENTS)
        self.linker_fragment_smiles = METAMIZOLE_SCAFFOLD_FRAGMENT

    def test_get_bond_atoms_idx_pairs(self):
        idx_pairs = self.bb.get_bond_atoms_idx_pairs(self.mol)
        self.assertEqual(len(idx_pairs), 2)
        self.assertEqual(len(idx_pairs[0]), 2)

    def test_get_linker_fragment(self):
        linker_smi = self.conversions.mol_to_smiles(self.bb.get_linker_fragment(self.mol))
        self.assertEqual(self.linker_fragment_smiles, linker_smi)

    def test_get_labeled_atom_dict(self):
        self.assertEqual(["0", "1"], list(self.bb.get_labeled_atom_dict(self.mol).keys()))

    def test_labeled_mol_into_fragment_mols(self):
        mol_fragments = self.bb.labeled_mol_into_fragment_mols(self.mol)
        self.assertEqual(len(mol_fragments), 3)
