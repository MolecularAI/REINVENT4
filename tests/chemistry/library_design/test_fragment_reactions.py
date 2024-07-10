import unittest

from reinvent.chemistry import conversions
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from tests.chemistry.library_design.fixtures import FRAGMENT_REACTION_SUZUKI, SCAFFOLD_SUZUKI
from tests.chemistry.fixtures.test_data import (
    CELECOXIB,
    ASPIRIN,
    CELECOXIB_FRAGMENT,
    METHYLPHEMYL_FRAGMENT,
)


class TestFragmentReactions(unittest.TestCase):
    def setUp(self):
        self.reactions = FragmentReactions()
        self._suzuki_reaction_dto_list = self.reactions.create_reactions_from_smirks(
            FRAGMENT_REACTION_SUZUKI
        )
        self.suzuki_positive_smile = CELECOXIB
        self.suzuki_negative_smile = ASPIRIN
        self.suzuki_fragment = SCAFFOLD_SUZUKI

    def test_slicing_molecule_to_fragments(self):
        molecule = conversions.smile_to_mol(self.suzuki_positive_smile)
        all_fragment_pairs = self.reactions.slice_molecule_to_fragments(
            molecule, self._suzuki_reaction_dto_list
        )
        smile_fragments = []
        for pair in all_fragment_pairs:
            smiles_pair = []

            for fragment in pair:
                smile = conversions.mol_to_smiles(fragment)
                smiles_pair.append(smile)
            smile_fragments.append(tuple(smiles_pair))

        self.assertEqual(METHYLPHEMYL_FRAGMENT, smile_fragments[0][0])
        self.assertEqual(CELECOXIB_FRAGMENT, smile_fragments[0][1])

    def test_slicing_wrong_molecule_to_fragments(self):
        molecule = conversions.smile_to_mol(self.suzuki_negative_smile)
        all_fragment_pairs = self.reactions.slice_molecule_to_fragments(
            molecule, self._suzuki_reaction_dto_list
        )
        smile_fragments = []
        for pair in all_fragment_pairs:
            smiles_pair = []
            for fragment in pair:
                smile = conversions.mol_to_smiles(fragment)
                smiles_pair.append(smile)
            smile_fragments.append(tuple(smiles_pair))
        self.assertEqual(0, len(smile_fragments))

    def test_slicing_suzuki_fragment(self):
        molecule = conversions.smile_to_mol(self.suzuki_fragment)
        all_fragment_pairs = self.reactions.slice_molecule_to_fragments(
            molecule, self._suzuki_reaction_dto_list
        )
        smile_fragments = []
        for pair in all_fragment_pairs:
            smiles_pair = []
            for fragment in pair:
                smile = conversions.mol_to_smiles(fragment)
                smiles_pair.append(smile)
            smile_fragments.append(tuple(smiles_pair))
        self.assertEqual(2, len(smile_fragments))
