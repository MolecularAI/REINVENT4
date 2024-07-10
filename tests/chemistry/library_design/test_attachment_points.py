import unittest

from rdkit.Chem.rdmolfiles import MolToSmiles

from reinvent.chemistry import conversions
from reinvent.chemistry.library_design import bond_maker, attachment_points
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from tests.chemistry.library_design.fixtures import (
    ANILINE_DERIVATIVE_DECORATIONS,
    ANOTHER_ANILINE_DERIVATIVE,
    LABELED_SLICED_ANILINE_DERIVATIVE,
    SLICED_ANILINE_DERIVATIVE,
)


class TestAttachmentPoints(unittest.TestCase):
    def setUp(self):
        self.reactions = FragmentReactions()
        self.decorations_A = ANILINE_DERIVATIVE_DECORATIONS
        self.expected_results = ANOTHER_ANILINE_DERIVATIVE
        self.labeled_scaffold_A = LABELED_SLICED_ANILINE_DERIVATIVE

    def test_get_attachment_points(self):
        result = attachment_points.get_attachment_points(self.labeled_scaffold_A)
        self.assertEqual([0, 2, 1], result)

    def test_add_attachment_point_numbers(self):
        relabeled = attachment_points.add_attachment_point_numbers(self.labeled_scaffold_A)
        result = attachment_points.get_attachment_points(relabeled)
        self.assertEqual([0, 1, 2], result)

    def test_remove_attachment_point_numbers(self):
        result = attachment_points.remove_attachment_point_numbers(self.labeled_scaffold_A)
        self.assertEqual(SLICED_ANILINE_DERIVATIVE, result)

    def test_remove_attachment_point_numbers_from_mol(self):
        molecule = conversions.smile_to_mol(self.labeled_scaffold_A)
        mol_result = attachment_points.remove_attachment_point_numbers_from_mol(molecule)
        result_no_brackets = MolToSmiles(mol_result, isomericSmiles=False, canonical=True)
        result = attachment_points.add_brackets_to_attachment_points(result_no_brackets)
        self.assertEqual("[*]c1ccc(N([*])[*])cc1", result)

    def test_add_brackets_to_attachment_points(self):
        result = attachment_points.add_brackets_to_attachment_points(self.decorations_A)
        self.assertEqual("[*]CCC|[*]CCC|[*]C", result)

    def test_add_first_attachment_point_number(self):
        result = attachment_points.add_first_attachment_point_number(SLICED_ANILINE_DERIVATIVE, 0)
        self.assertEqual("[*:0]N([*])c1ccc(cc1)[*]", result)

    def test_get_attachment_points_for_molecule(self):
        molecule = conversions.smile_to_mol(self.labeled_scaffold_A)
        result = attachment_points.get_attachment_points_for_molecule(molecule)
        self.assertEqual([0, 2, 1], result)
