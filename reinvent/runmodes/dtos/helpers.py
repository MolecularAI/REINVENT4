"""Helper DTO for chemistry"""

from dataclasses import dataclass

from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import BondMaker, AttachmentPoints


@dataclass
class ChemistryHelpers:
    conversions: Conversions
    bond_maker: BondMaker
    attachment_points: AttachmentPoints
