import unittest

from rdkit import Chem

from reinvent.chemistry.library_design.reaction_filters import ReactionFiltersEnum
from tests.chemistry.fixtures.test_data import (
    REACTION_SUZUKI,
    SCAFFOLD_TO_DECORATE,
    TWO_DECORATIONS_SUZUKI,
    TWO_DECORATIONS_ONE_SUZUKI,
)
from tests.chemistry.library_design.reaction_filters.base_reaction_filter import BaseTestReactionFilter


class TestSelectiveReactionFilter(BaseTestReactionFilter):
    def setUp(self):
        self.type = ReactionFiltersEnum().SELECTIVE
        self.reactions = [[REACTION_SUZUKI], [REACTION_SUZUKI]]
        super().setUp()

    @unittest.skip("non-functional in R4")
    def test_two_attachment_points_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_TO_DECORATE
        decoration = TWO_DECORATIONS_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    @unittest.skip("non-functional in R4")
    def test_two_attachment_points_one_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_TO_DECORATE
        decoration = TWO_DECORATIONS_ONE_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.75, score)
