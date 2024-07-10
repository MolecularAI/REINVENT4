import unittest

from rdkit import Chem

from reinvent.chemistry.library_design import bond_maker, attachment_points
from reinvent.chemistry.library_design.reaction_filters import ReactionFiltersEnum
from reinvent.chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter
from reinvent.chemistry.library_design.reaction_filters.reaction_filter_configruation import (
    ReactionFilterConfiguration,
)
from tests.chemistry.fixtures import default_reaction_definitions
from tests.chemistry.fixtures.test_data import (
    DECORATION_SUZUKI,
    SCAFFOLD_SUZUKI,
    SCAFFOLD_NO_SUZUKI,
    DECORATION_NO_SUZUKI,
    SCAFFOLD_TO_DECORATE,
    TWO_DECORATIONS_SUZUKI,
    TWO_DECORATIONS_ONE_SUZUKI,
    REACTION_SUZUKI_NAME,
)


class TestDefinedSelectiveFilterSingleReaction(unittest.TestCase):
    def setUp(self):
        self._enum = ReactionFiltersEnum()
        reactions = [[REACTION_SUZUKI_NAME]]
        with default_reaction_definitions() as reaction_definitions_path:
            configuration = ReactionFilterConfiguration(
                type=self._enum.DEFINED_SELECTIVE,
                reactions=reactions,
                reaction_definition_file=str(reaction_definitions_path),
            )
        self.reaction_filter = ReactionFilter(configuration)

    def test_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_SUZUKI
        decoration = DECORATION_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    def test_with_non_suzuki_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.0, score)


class TestDefinedSelectiveFilter(unittest.TestCase):
    def setUp(self):
        self._enum = ReactionFiltersEnum()
        reactions = [[REACTION_SUZUKI_NAME], [REACTION_SUZUKI_NAME]]
        with default_reaction_definitions() as reaction_definitions_path:
            configuration = ReactionFilterConfiguration(
                type=self._enum.DEFINED_SELECTIVE,
                reactions=reactions,
                reaction_definition_file=str(reaction_definitions_path),
            )

        self.reaction_filter = ReactionFilter(configuration)

    def test_two_attachment_points_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_TO_DECORATE
        decoration = TWO_DECORATIONS_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    def test_two_attachment_points_one_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_TO_DECORATE
        decoration = TWO_DECORATIONS_ONE_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.0, score)
