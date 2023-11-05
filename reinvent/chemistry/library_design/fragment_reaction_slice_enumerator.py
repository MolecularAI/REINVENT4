from collections import OrderedDict
from typing import List, Tuple, Set

from rdkit.Chem.rdchem import Mol

from reinvent.chemistry import TransformationTokens, Conversions
from reinvent.chemistry.library_design import FragmentFilter
from reinvent.chemistry.library_design.dtos import FilteringConditionDTO, ReactionDTO
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from reinvent.chemistry.library_design.fragmented_molecule import FragmentedMolecule


class FragmentReactionSliceEnumerator:
    def __init__(
        self,
        chemical_reactions: List[ReactionDTO],
        scaffold_conditions: List[FilteringConditionDTO],
        decoration_conditions: List[FilteringConditionDTO],
    ):
        """
        Class to enumerate slicings given certain conditions.
        :param chemical_reactions: A list of ChemicalReaction objects.
        :param scaffold_conditions: Conditions to use when filtering scaffolds obtained from slicing molecules (see FragmentFilter).
        :param decoration_conditions: Conditions to use when filtering decorations obtained from slicing molecules.
        """
        self._tockens = TransformationTokens()
        self._chemical_reactions = chemical_reactions
        self._scaffold_filter = FragmentFilter(scaffold_conditions)
        self._decoration_filter = FragmentFilter(decoration_conditions)
        self._reactions = FragmentReactions()
        self._conversions = Conversions()

    def enumerate(self, molecule: Mol, cuts: int) -> List[FragmentedMolecule]:
        """
        Enumerates all possible combination of slicings of a molecule given a number of cuts.
        :param molecule: A mol object with the molecule to slice.
        :param cuts: The number of cuts to perform.
        :return : A list with all the possible (scaffold, decorations) pairs as SlicedMol objects.
        """
        original_smiles = self._conversions.mol_to_smiles(molecule)
        sliced_mols = set()
        for cut in range(1, cuts + 1):
            if cut == 1:
                fragment_pairs = self._reactions.slice_molecule_to_fragments(
                    molecule, self._chemical_reactions
                )

                for pair in fragment_pairs:
                    for indx, _ in enumerate(pair):
                        decorations = self._select_all_except(pair, indx)
                        decoration = self._conversions.copy_mol(decorations[0])
                        labeled_decoration = OrderedDict()
                        labeled_decoration[0] = decoration  # [ for decoration in decorations]

                        scaffold = self._conversions.copy_mol(pair[indx])
                        labeled_scaffold = self._label_scaffold(scaffold)

                        # TODO: filtering should take place after scaffold is generated
                        sliced_mol = FragmentedMolecule(
                            labeled_scaffold, labeled_decoration, original_smiles
                        )
                        if sliced_mol.original_smiles == sliced_mol.reassembled_smiles:
                            sliced_mols.add(sliced_mol)
            else:
                for slice in sliced_mols:
                    to_add = self._scaffold_slicing(slice, cut)
                    sliced_mols = sliced_mols.union(to_add)

        return list(filter(self._filter, sliced_mols))

    def _scaffold_slicing(self, slice: FragmentedMolecule, cut: int) -> Set[FragmentedMolecule]:
        to_add = set()
        if slice.decorations_count() == cut - 1:
            fragment_pairs = self._reactions.slice_molecule_to_fragments(
                slice.scaffold, self._chemical_reactions
            )

            for pair in fragment_pairs:
                scaffold, decoration = self._split_scaffold_from_decorations(pair, cut)
                if scaffold:
                    labeled_scaffold = self._label_scaffold(scaffold)
                    labeled_scaffold = self._conversions.copy_mol(labeled_scaffold)
                    decoration = self._conversions.copy_mol(decoration)
                    sliced_mol = self._create_sliced_molecule(slice, labeled_scaffold, decoration)

                    if sliced_mol.original_smiles == sliced_mol.reassembled_smiles:
                        to_add.add(sliced_mol)
        return to_add

    def _select_all_except(self, fragments: Tuple[Mol], to_exclude: int) -> List[Mol]:
        return [fragment for indx, fragment in enumerate(fragments) if indx != to_exclude]

    def _filter(self, sliced_mol: FragmentedMolecule) -> bool:
        return self._scaffold_filter.filter(sliced_mol.scaffold) and all(
            self._decoration_filter.filter(dec) for dec in sliced_mol.decorations.values()
        )

    def _split_scaffold_from_decorations(self, pair: Tuple[Mol], cuts: int) -> Tuple[Mol, Mol]:
        decoration = None
        scaffold = None
        for frag in pair:
            num_att = len(
                [
                    atom
                    for atom in frag.GetAtoms()
                    if atom.GetSymbol() == self._tockens.ATTACHMENT_POINT_TOKEN
                ]
            )
            # detect whether there is one fragment with as many attachment points as cuts (scaffold)
            # the rest are decorations
            if num_att == cuts and not scaffold:
                scaffold = frag
            if num_att == 1:
                decoration = frag
        if decoration and scaffold:
            return scaffold, decoration
        else:
            return (None, None)

    def _label_scaffold(self, scaffold: Mol) -> Mol:
        highest_number = self._find_highest_number(scaffold)

        for atom in scaffold.GetAtoms():
            if atom.GetSymbol() == self._tockens.ATTACHMENT_POINT_TOKEN:
                try:
                    atom_number = int(atom.GetProp("molAtomMapNumber"))
                except:
                    highest_number += 1
                    num = atom.GetIsotope()
                    atom.SetIsotope(0)
                    atom.SetProp("molAtomMapNumber", str(highest_number))
        scaffold.UpdatePropertyCache()

        return scaffold

    def _find_highest_number(self, cut_mol: Mol) -> int:
        highest_number = -1

        for atom in cut_mol.GetAtoms():
            if atom.GetSymbol() == self._tockens.ATTACHMENT_POINT_TOKEN:
                try:
                    atom_number = int(atom.GetProp("molAtomMapNumber"))
                    if highest_number < atom_number:
                        highest_number = atom_number
                except:
                    pass
        return highest_number

    def _create_sliced_molecule(
        self, original_sliced_mol: FragmentedMolecule, scaffold: Mol, decoration: Mol
    ) -> FragmentedMolecule:
        old_decorations = OrderedDict()
        for k, v in original_sliced_mol.decorations.items():
            old_decorations[k] = v
        old_decorations[original_sliced_mol.decorations_count()] = decoration
        sliced_mol = FragmentedMolecule(
            scaffold, old_decorations, original_sliced_mol.original_smiles
        )
        return sliced_mol
