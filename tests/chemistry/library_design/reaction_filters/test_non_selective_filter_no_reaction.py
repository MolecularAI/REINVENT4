from rdkit import Chem

from reinvent.chemistry.library_design import bond_maker, attachment_points
from reinvent.chemistry.library_design.reaction_filters import ReactionFiltersEnum
from tests.chemistry.fixtures.test_data import (
    SCAFFOLD_SUZUKI,
    DECORATION_SUZUKI,
    SCAFFOLD_NO_SUZUKI,
    DECORATION_NO_SUZUKI,
)
from tests.chemistry.library_design.reaction_filters.base_reaction_filter import (
    BaseTestReactionFilter,
)


class TestNonSelectiveReactionFiltersNoReaction(BaseTestReactionFilter):
    def setUp(self):
        self.type = ReactionFiltersEnum().NON_SELECTIVE
        self.reactions = [[]]
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

    def test_with_any_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule: Chem.Mol = bond_maker.join_scaffolds_and_decorations(
            scaffold, decoration, keep_labels_on_atoms=True
        )
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)
