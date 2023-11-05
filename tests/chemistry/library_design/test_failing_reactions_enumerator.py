import unittest

from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import FailingReactionsEnumerator
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from tests.chemistry.fixtures.test_data import CELECOXIB
from tests.chemistry.library_design.fixtures import FRAGMENT_REACTIONS, FRAGMENT_REACTION_SUZUKI


class TestFailingReactionsEnumerator(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        self.reactions = FragmentReactions()
        self._suzuki_reaction_dto_list = self.reactions.create_reactions_from_smirks(
            FRAGMENT_REACTIONS
        )
        self.suzuki_positive_smile = CELECOXIB
        self.suzuki_positive_molecule = self.chemistry.smile_to_mol(self.suzuki_positive_smile)

        self._slice_enumerator = FailingReactionsEnumerator(self._suzuki_reaction_dto_list)

    def test_failed_slicing_1(self):
        result = self._slice_enumerator.enumerate(self.suzuki_positive_molecule, 1)

        self.assertEqual(1, len(result))
        self.assertEqual(
            result[0].reaction_smirks,
            "[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][c:2]>>[*:2][*].[*:1][*]",
        )

    def test_failed_slicing_2(self):
        result = self._slice_enumerator.enumerate(self.suzuki_positive_molecule, 3)

        self.assertEqual(2, len(result))
        self.assertEqual(
            result[0].reaction_smirks,
            "[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][c:2]>>[*:2][*].[*:1][*]",
        )


class TestNonFailingReactionsEnumerator(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        self.reactions = FragmentReactions()
        self._suzuki_reaction_dto_list = self.reactions.create_reactions_from_smirks(
            FRAGMENT_REACTION_SUZUKI
        )
        self.suzuki_positive_smile = CELECOXIB
        self.suzuki_positive_molecule = self.chemistry.smile_to_mol(self.suzuki_positive_smile)

        self._slice_enumerator = FailingReactionsEnumerator(self._suzuki_reaction_dto_list)

    def test_non_failed_slicing_1(self):
        result = self._slice_enumerator.enumerate(self.suzuki_positive_molecule, 1)

        self.assertEqual(0, len(result))
