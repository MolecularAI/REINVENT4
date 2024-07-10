from typing import List, Dict, Tuple

from rdkit.Chem.rdchem import Mol, AtomKekulizeException, Atom
from rdkit.Chem.rdmolops import FragmentOnBonds, GetMolFrags

from reinvent.chemistry import conversions, tokens
from reinvent.chemistry.library_design.dtos import ReactionOutcomeDTO


class BondMapper:
    def convert_building_blocks_to_fragments(
        self, molecule: Mol, neighbor_map: Dict, list_of_outcomes: List[ReactionOutcomeDTO]
    ) -> List[Tuple[Mol]]:
        all_fragments = []

        for outcome in list_of_outcomes:
            try:
                reagent_pairs = outcome.reaction_outcomes

                for reagent_pair in reagent_pairs:
                    list_of_atom_pairs = self._find_bonds_targeted_by_reaction(
                        reagent_pair, neighbor_map
                    )
                    bonds_to_cut = self._find_indices_of_target_bonds(molecule, list_of_atom_pairs)

                    if len(bonds_to_cut) == 1:
                        reaction_fragments = self._create_fragments(molecule, bonds_to_cut)
                        all_fragments.append(reaction_fragments)
            except AtomKekulizeException as ex:
                raise AtomKekulizeException(
                    f"failed scaffold: {conversions.mol_to_smiles(molecule)} \n for reaction: {outcome.reaction_smarts} \n {ex}"
                ) from ex
        return all_fragments

    def _find_bonds_targeted_by_reaction(
        self, reagent_pair: Tuple[Mol], neighbor_map: Dict
    ) -> List[Tuple[int]]:
        atom_pairs = []
        for reagent in reagent_pair:
            reactant_map = self._create_neighbor_map_for_reactant(reagent)
            atom_pair = self._indentify_mismatching_indices(neighbor_map, reactant_map)
            atom_pairs.extend(atom_pair)
        return atom_pairs

    def _create_neighbor_map_for_reactant(self, reactant: Mol) -> Dict[int, List[int]]:
        interaction_map = {}

        for atom in reactant.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                neighbor_indxs = self._get_original_ids_from_reactant(atom)
                interaction_map[int(atom.GetProp("react_atom_idx"))] = neighbor_indxs

        return interaction_map

    def _get_original_ids_from_reactant(self, atom: Atom) -> List[int]:
        neighbours = atom.GetNeighbors()
        indices = [
            int(neighbor.GetProp("react_atom_idx"))
            for neighbor in neighbours
            if neighbor.HasProp("react_atom_idx")
        ]
        neighbor_indxs = [idx for idx in indices]

        return neighbor_indxs

    def _indentify_mismatching_indices(self, original: Dict, derived: Dict) -> List[Tuple]:
        def is_a_mismatch(original_points: [], derived_points: []):
            original_points.sort()
            derived_points.sort()
            return original_points != derived_points

        mismatching_indices = []

        for key in derived.keys():
            if is_a_mismatch(original.get(key), derived.get(key)):
                differences = list(set(original.get(key)) - set(derived.get(key)))
                for difference in differences:
                    pair = (key, difference)
                    mismatching_indices.append(tuple(sorted(pair)))

        return mismatching_indices

    def _find_indices_of_target_bonds(
        self, molecule: Mol, list_of_atom_pairs: List[Tuple[int]]
    ) -> List[int]:
        list_of_atom_pairs = list(set(list_of_atom_pairs))
        bonds_to_cut = [
            molecule.GetBondBetweenAtoms(pair[0], pair[1]).GetIdx() for pair in list_of_atom_pairs
        ]
        return bonds_to_cut

    def _create_fragments(self, molecule: Mol, bonds_to_cut: List[int]) -> Tuple[Mol]:
        attachment_point_idxs = [(i, i) for i in range(len(bonds_to_cut))]
        cut_mol = FragmentOnBonds(
            molecule, bondIndices=bonds_to_cut, dummyLabels=attachment_point_idxs
        )

        for atom in cut_mol.GetAtoms():
            if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN:
                num = atom.GetIsotope()
                atom.SetIsotope(0)
                atom.SetProp("molAtomMapNumber", str(num))
        cut_mol.UpdatePropertyCache()
        fragments = GetMolFrags(cut_mol, asMols=True, sanitizeFrags=True)

        return fragments
