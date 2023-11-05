import unittest

from reinvent.chemistry.library_design import BondMaker, AttachmentPoints

from reinvent.chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter
from reinvent.chemistry.library_design.reaction_filters.reaction_filter_configruation import (
    ReactionFilterConfiguration,
)


class BaseTestReactionFilter(unittest.TestCase):
    def setUp(self):
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        configuration = ReactionFilterConfiguration(type=self.type, reactions=self.reactions)
        self.reaction_filter = ReactionFilter(configuration)
