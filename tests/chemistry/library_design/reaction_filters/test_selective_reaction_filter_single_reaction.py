import unittest

from rdkit import Chem

from reinvent.chemistry.library_design import bond_maker, attachment_points
from reinvent.chemistry.library_design.reaction_filters import ReactionFiltersEnum
from tests.chemistry.fixtures.test_data import (
    REACTION_SUZUKI,
    SCAFFOLD_SUZUKI,
    SCAFFOLD_NO_SUZUKI,
    DECORATION_NO_SUZUKI,
    DECORATION_SUZUKI,
)
from tests.chemistry.library_design.reaction_filters.base_reaction_filter import (
    BaseTestReactionFilter,
)


class TestSelectiveReactionFilterSingleReaction(BaseTestReactionFilter):
    def setUp(self):
        self.type = ReactionFiltersEnum().SELECTIVE
        self.reactions = [[REACTION_SUZUKI]]
        super().setUp()

    def test_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_SUZUKI
        decoration = DECORATION_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    def test_with_non_suzuki_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.0, score)
