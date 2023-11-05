import unittest

from rdkit import Chem

from reinvent.chemistry.library_design import BondMaker, AttachmentPoints
from reinvent.chemistry.library_design.reaction_filters import ReactionFiltersEnum
from reinvent.chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter
from reinvent.chemistry.library_design.reaction_filters.reaction_filter_configruation import (
    ReactionFilterConfiguration,
)
from tests.chemistry.fixtures import default_reaction_definitions
from tests.chemistry.fixtures.test_data import (
    REACTION_SUZUKI_NAME,
    SCAFFOLD_SUZUKI,
    DECORATION_SUZUKI,
    SCAFFOLD_NO_SUZUKI,
    DECORATION_NO_SUZUKI,
)


class TestDefinedSelectiveFilterSingleReaction(unittest.TestCase):
    def setUp(self):
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._enum = ReactionFiltersEnum()
        reactions = [[REACTION_SUZUKI_NAME]]
        with default_reaction_definitions() as reaction_definitions_path:
            configuration = ReactionFilterConfiguration(
                type=self._enum.DEFINED_SELECTIVE,
                reactions=reactions,
                reaction_definition_file=str(reaction_definitions_path),
            )
        self.reaction_filter = ReactionFilter(configuration)

    @unittest.skip("non-functional in R4")
    def test_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_SUZUKI
        decoration = DECORATION_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    @unittest.skip("non-functional in R4")
    def test_with_non_suzuki_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.5, score)
