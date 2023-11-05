from collections import OrderedDict
from typing import List, Tuple

from rdkit.Chem.rdchem import Mol

from reinvent.chemistry import TransformationTokens, Conversions
from reinvent.chemistry.library_design.dtos import ReactionDTO, FailedReactionDTO
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from reinvent.chemistry.library_design.fragmented_molecule import FragmentedMolecule


class FailingReactionsEnumerator:
    def __init__(self, chemical_reactions: List[ReactionDTO]):
        """
        Class to enumerate over a list of reactions and a molecule in order to detect failures.
        :param chemical_reactions: A list of ReactionDTO objects.
        """
        self._tockens = TransformationTokens()
        self._chemical_reactions = chemical_reactions

        self._reactions = FragmentReactions()
        self._conversions = Conversions()

    def enumerate(self, molecule: Mol, failures_limit: int) -> List[FailedReactionDTO]:
        """
        Enumerates all provided reactions on a molecule in order detect failures.
        :param molecule: A mol object with the molecule to apply reactions on.
        :param failures_limit: The number of failed examples to accumulate.
        :return : A list of failed reaction/molecule pairs.
        """
        original_smiles = self._conversions.mol_to_smiles(molecule)
        failed_reactions = {}

        for reaction in self._chemical_reactions:
            dto = self._reactions.apply_reaction_on_molecule(molecule, reaction)

            for pair in dto.reaction_outcomes:
                if failures_limit <= len(failed_reactions):
                    break
                for indx, _ in enumerate(pair):
                    decorations = self._select_all_except(pair, indx)
                    decoration = self._conversions.copy_mol(decorations[0])
                    scaffold = self._conversions.copy_mol(pair[indx])

                    if not self._validate(scaffold, decoration, original_smiles):
                        failed_reaction = FailedReactionDTO(
                            reaction.reaction_smarts, original_smiles
                        )
                        failed_reactions[failed_reaction.reaction_smirks] = failed_reaction
                        break

            if failures_limit <= len(failed_reactions):
                break

        dtos = [dto for dto in failed_reactions.values()]

        return dtos

    def _select_all_except(self, fragments: Tuple[Mol], to_exclude: int) -> List[Mol]:
        return [fragment for indx, fragment in enumerate(fragments) if indx != to_exclude]

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

    def _validate(self, scaffold: Mol, decoration: Mol, original_smiles: str) -> bool:
        if scaffold and decoration:
            labeled_decoration = OrderedDict()
            labeled_decoration[0] = decoration
            labeled_scaffold = self._label_scaffold(scaffold)
            sliced_mol = FragmentedMolecule(labeled_scaffold, labeled_decoration, original_smiles)

            if original_smiles != sliced_mol.reassembled_smiles:
                return False
            return True
        else:
            return False
