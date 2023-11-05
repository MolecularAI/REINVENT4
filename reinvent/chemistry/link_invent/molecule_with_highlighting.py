import re
import io
from typing import List, Tuple
from collections import defaultdict

from matplotlib import cm
from numpy import linspace
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

from reinvent.chemistry import Conversions, TransformationTokens
from reinvent.chemistry.library_design import BondMaker, AttachmentPoints
from reinvent.chemistry.link_invent.bond_breaker import BondBreaker


class MoleculeWithHighlighting:
    def __init__(self, color_map_name: str = "Set3", image_size=(400, 400)):
        self._conversions = Conversions()
        self._tokens = TransformationTokens()
        self._bond_maker = BondMaker()
        self._bond_breaker = BondBreaker()
        self._attachment_points = AttachmentPoints()

        self._color_map_name = color_map_name  # name of the matplotlib colormap
        self._image_size = image_size

        self._re_attachment_point = re.compile(self._tokens.ATTACHMENT_POINT_REGEXP)

    def get_image(self, mol: Mol, parts_str_list: List[str], label: str):
        mol = self._make_mole_canonical(mol)
        atom_dict, bond_dict = self._get_highlight_dicts(parts_str_list)

        d = rdMolDraw2D.MolDraw2DCairo(*self._image_size)
        d.DrawMoleculeWithHighlights(mol, label, atom_dict, bond_dict, {}, {})
        image = Image.open(io.BytesIO(d.GetDrawingText()))
        return image

    def _get_highlight_dicts(self, parts_str_list: List[str]) -> Tuple[dict, dict]:
        atoms_to_highlight = []
        bonds_to_highlight = []
        for idx, parts_str in enumerate(parts_str_list):
            linker, warhead = self._get_parts(parts_str)
            mol_numbered_ap = self._get_labeled_mol(linker, warhead)
            mol_numbered_ap = self._make_mole_canonical(mol_numbered_ap)
            atom_pairs_to_highlight = self._bond_breaker.get_bond_atoms_idx_pairs(mol_numbered_ap)
            bonds_to_highlight.append(
                [
                    mol_numbered_ap.GetBondBetweenAtoms(*atom_pair).GetIdx()
                    for atom_pair in atom_pairs_to_highlight
                ]
            )
            atoms_to_highlight.append(
                [item for sublist in atom_pairs_to_highlight for item in sublist]
            )

        n_colors = len(parts_str_list)
        colors = [tuple(c) for c in getattr(cm, self._color_map_name)(linspace(0, 1, n_colors))]
        atom_dict = self._get_color_dict(atoms_to_highlight, colors)
        bond_dict = self._get_color_dict(bonds_to_highlight, colors)
        return atom_dict, bond_dict

    def _get_parts(self, smile_parts_str: str) -> Tuple[str, str]:
        parts_list = smile_parts_str.split(self._tokens.ATTACHMENT_SEPARATOR_TOKEN)
        n_ap = [len(self._re_attachment_point.findall(p)) for p in parts_list]
        linker = [part for part, number_ap in zip(parts_list, n_ap) if number_ap == 2][0]
        warheads = self._tokens.ATTACHMENT_SEPARATOR_TOKEN.join(
            [part for part, number_ap in zip(parts_list, n_ap) if number_ap == 1]
        )
        return linker, warheads

    def _get_labeled_mol(self, linker_smi: str, warheads_smi: str) -> Mol:
        linker_numbered = self._attachment_points.add_attachment_point_numbers(
            linker_smi, canonicalize=False
        )
        mol = self._bond_maker.join_scaffolds_and_decorations(
            linker_numbered, warheads_smi, keep_labels_on_atoms=True
        )
        return mol

    def _make_mole_canonical(self, mol: Mol) -> Mol:
        return self._conversions.smile_to_mol(self._conversions.mol_to_smiles(mol, canonical=True))

    @staticmethod
    def _get_color_dict(index_list: List[List[int]], color_list: List[tuple]) -> dict:
        color_dict = defaultdict(list)
        for color_idx, indexes in enumerate(index_list):
            for index in indexes:
                color_dict[index].append(color_list[color_idx])
        return dict(color_dict)
