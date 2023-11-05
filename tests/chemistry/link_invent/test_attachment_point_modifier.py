import unittest

from rdkit.Chem import Mol

from reinvent.chemistry import Conversions
from reinvent.chemistry.link_invent.bond_breaker import BondBreaker
from reinvent.chemistry.link_invent.attachment_point_modifier import AttachmentPointModifier
from tests.chemistry.fixtures.test_data import (
    METAMIZOLE_LABELED_COMPONENTS,
    METAMIZOLE_SCAFFOLD_FRAGMENT,
    DIMETHYL_AMINO_PYRAZOLE,
    LINKER_THREE_SQUARE_BRACKETS,
    LINKER_CHARGED_ATTACHMENT_ATOM,
)


class TestAttachmentPointModifier(unittest.TestCase):
    def setUp(self) -> None:
        self.conversions = Conversions()
        self.bond_breaker = BondBreaker()
        self.attachment_point_modifier = AttachmentPointModifier()
        self.mol = self.conversions.smile_to_mol(METAMIZOLE_LABELED_COMPONENTS)
        self.linker_fragment_smiles = METAMIZOLE_SCAFFOLD_FRAGMENT
        self.linker_three_square_brackets = LINKER_THREE_SQUARE_BRACKETS
        self.linker_charged_attachment_atom = LINKER_CHARGED_ATTACHMENT_ATOM

    def test_extract_attachment_atoms(self):
        attachment_atoms = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_fragment_smiles
        )
        self.assertEqual(len(attachment_atoms), 2)
        self.assertEqual(attachment_atoms, ["CH2:1", "n:0"])

        selected_attachment_atoms = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_three_square_brackets
        )
        self.assertEqual(len(selected_attachment_atoms), 2)
        self.assertEqual(selected_attachment_atoms, ["c:0", "CH:1"])

    def test_add_explicit_H_to_atom(self):
        atom1 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_fragment_smiles
        )[0]
        atom2 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_fragment_smiles
        )[1]
        atom1_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom1)
        atom2_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom2)
        self.assertEqual(atom1_modified, "CH3")
        self.assertEqual(atom2_modified, "nH")

        atom3 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_three_square_brackets
        )[0]
        atom4 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_three_square_brackets
        )[1]
        atom3_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom3)
        atom4_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom4)
        self.assertEqual(atom3_modified, "cH")
        self.assertEqual(atom4_modified, "CH2")

        atom5 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_charged_attachment_atom
        )[0]
        atom6 = self.attachment_point_modifier.extract_attachment_atoms(
            self.linker_charged_attachment_atom
        )[1]
        atom5_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom5)
        atom6_modified = self.attachment_point_modifier.add_explicit_H_to_atom(atom6)
        self.assertEqual(atom5_modified, "N+")
        self.assertEqual(atom6_modified, "NH2")

    def test_cap_linker_with_hydrogen(self):
        linker_mol = self.bond_breaker.get_linker_fragment(self.mol)
        capped_linker_mol = self.attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        capped_linker_smiles = self.conversions.mol_to_smiles(capped_linker_mol)

        self.assertEqual(capped_linker_smiles, DIMETHYL_AMINO_PYRAZOLE)
        self.assertIsInstance(capped_linker_mol, Mol)
