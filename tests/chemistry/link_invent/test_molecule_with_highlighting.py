import unittest

from PIL import Image
from reinvent.chemistry import Conversions, TransformationTokens

from reinvent.chemistry.link_invent.molecule_with_highlighting import MoleculeWithHighlighting
from tests.chemistry.fixtures.test_data import METAMIZOLE, METAMIZOLE_SCAFFOLD, METAMIZOLE_DECORATIONS


class TestMoleculeWithHighlighting(unittest.TestCase):
    def setUp(self) -> None:
        self.conversions = Conversions()
        self.tokens = TransformationTokens()
        self.label = "metamizole"
        self.mol = self.conversions.smile_to_mol(METAMIZOLE)
        self.parts_list = [
            self.tokens.ATTACHMENT_SEPARATOR_TOKEN.join(
                [METAMIZOLE_SCAFFOLD, METAMIZOLE_DECORATIONS]
            )
        ]
        self.parts_list_2 = [
            self.tokens.ATTACHMENT_SEPARATOR_TOKEN.join(
                [METAMIZOLE_DECORATIONS, METAMIZOLE_SCAFFOLD]
            )
        ]
        self.molecule_with_highlighting = MoleculeWithHighlighting()

    def test_single_parts_link_invent(self):
        img = self.molecule_with_highlighting.get_image(self.mol, self.parts_list, self.label)
        img_2 = self.molecule_with_highlighting.get_image(self.mol, self.parts_list_2, self.label)
        self.assertTrue(isinstance(img, Image.Image))
        self.assertTrue(isinstance(img_2, Image.Image))

    def test_multi_parts_link_invent(self):
        img = self.molecule_with_highlighting.get_image(self.mol, self.parts_list * 3, self.label)
        self.assertTrue(isinstance(img, Image.Image))
