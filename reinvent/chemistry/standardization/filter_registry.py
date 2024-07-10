from typing import List
import logging

from rdkit.Chem import AllChem, MolToSmiles
from rdkit.Chem import SaltRemover
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles
from rdkit.Chem.rdmolops import RemoveHs

from reinvent.chemistry.standardization.filter_types_enum import FilterTypesEnum
from reinvent.models.reinvent.models.vocabulary import split_by, REGEXP_ORDER

logger = logging.getLogger(__name__)

# The classical Reinvent prior supports the following elements
# C, N, O, F, S, Cl, Br
DEFAULT_ELEMENTS = [6, 7, 8, 9, 16, 17, 35]

NEUTRALIZE_PATTERNS = (
    ("[n+;H]", "n"),  # Imidazoles
    ("[N+;!H0]", "N"),  # Amines
    ("[$([O-]);!$([O-][#7])]", "O"),  # Carboxylic acids and alcohols
    ("[S-;X1]", "S"),  # Thiols
    ("[$([N-;X2]S(=O)=O)]", "N"),  # Sulfonamides
    ("[$([N-;X2][C,N]=C)]", "N"),  # Enamines
    ("[n-]", "[nH]"),  # Tetrazoles
    ("[$([S-]=O)]", "S"),  # Sulfoxides
    ("[$([N-]C=O)]", "N"),  # Amides
)


class FilterRegistry:
    def __init__(self):
        filter_types = FilterTypesEnum()

        self._filters = {
            filter_types.NEUTRALIZE_CHARGES: self._neutralise_charges,
            filter_types.GET_LARGEST_FRAGMENT: self._get_largest_fragment,
            filter_types.REMOVE_HYDROGENS: self._remove_hydrogens,
            filter_types.REMOVE_SALTS: self._remove_salts,
            filter_types.GENERAL_CLEANUP: self._general_cleanup,
            filter_types.UNWANTED_PATTERNS: self._unwanted_patterns,
            filter_types.VOCABULARY_FILTER: self._vocabulary_filters,
            filter_types.VALID_SIZE: self._valid_size,
            filter_types.HEAVY_ATOM_FILTER: self._heavy_atom_filter,
            filter_types.ALLOWED_ELEMENTS: self._allowed_elements,
            filter_types.DEFAULT: self.standardize,
        }

    def get_filter(self, filter_name: str):
        selected_filter = None

        try:
            selected_filter = self._filters.get(filter_name)
        except:
            KeyError(f'requested filter "{filter_name}" does not exist in the registry')
        return selected_filter

    def _get_largest_fragment(self, mol):
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        maxmol = None

        for mol in frags:
            if mol is None:
                continue

            if maxmol is None:
                maxmol = mol

            if maxmol.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
                maxmol = mol
        return maxmol

    def _remove_hydrogens(self, mol):
        return RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)

    def _remove_salts(self, mol):
        return SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)

    def _neutralise_charges(self, mol, reactions=None):
        if reactions is None:
            reactions = [
                (MolFromSmarts(x), MolFromSmiles(y, False)) for x, y in NEUTRALIZE_PATTERNS
            ]

        for i, (reactant, product) in enumerate(reactions):
            while mol.HasSubstructMatch(reactant):
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]

        return mol

    def _general_cleanup(self, mol):
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)

        return mol

    def _unwanted_patterns(self, mol):
        if mol:
            cyano_filter = "[C-]#[N+]"
            oh_filter = "[OH+]"
            sulfur_filter = "[SH]"

            if (
                not mol.HasSubstructMatch(MolFromSmarts(cyano_filter))
                and not mol.HasSubstructMatch(MolFromSmarts(oh_filter))
                and not mol.HasSubstructMatch(MolFromSmarts(sulfur_filter))
            ):
                return mol

    def _vocabulary_filters(self, mol, vocabulary: List[str]):
        if mol:
            smiles = MolToSmiles(mol, isomericSmiles=False, canonical=True)
            tokens = split_by(smiles, REGEXP_ORDER)

            for token in tokens:
                if token not in vocabulary:
                    return None

            return mol

    def _allowed_elements(self, mol, elements=None):
        if elements is None:
            elements = DEFAULT_ELEMENTS

        if mol:
            valid_elements = all([atom.GetAtomicNum() in elements for atom in mol.GetAtoms()])

            if valid_elements:
                return mol

    def _heavy_atom_filter(self, mol, min_heavy_atoms=2, max_heavy_atoms=70):
        if mol:
            correct_size = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
            if correct_size:
                return mol

    def _valid_size(
        self,
        mol,
        min_heavy_atoms=2,
        max_heavy_atoms=70,
        element_list=None,
    ):
        """Filters molecules on valid elements and number of heavy atoms"""

        if mol:
            correct_size = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms

            if not correct_size:
                logger.debug(f"valid_size: {correct_size=}")
                return

            valid_elements = self._allowed_elements(mol, element_list)

            if not valid_elements:
                logger.debug(f"valid_size: {valid_elements=}")
                return

            return mol

    def standardize(
        self,
        mol,
        min_heavy_atoms=2,
        max_heavy_atoms=70,
        element_list=None,
        neutralise_charges=True,
    ):
        stage = ""

        if mol:
            mol = self._get_largest_fragment(mol)
            stage = "largest_fragment"
        if mol:
            mol = self._remove_hydrogens(mol)
            stage = "remove_hydrogens"
        if mol:
            mol = self._remove_salts(mol)
            stage = "remove_salts"
        if mol and neutralise_charges:
            mol = self._neutralise_charges(mol)
            stage = "neutralise_charges"
        if mol:
            mol = self._general_cleanup(mol)
            stage = "general_cleanup"
        if mol:
            mol = self._unwanted_patterns(mol)
            stage = "token_filters"
        if mol:
            mol = self._valid_size(mol, min_heavy_atoms, max_heavy_atoms, element_list)
            stage = "valid_size"

        if mol:
            return mol

        logger.debug(f"{__name__}: failed in stage {stage}")

        return None
