from typing import List, Tuple

from rdkit.Chem import AllChem, Mol
from rdkit.Chem.Lipinski import RingCount
from rdkit.Chem.rdChemReactions import ChemicalReaction

from reinvent.chemistry import conversions
from reinvent.chemistry.library_design import BondMapper
from reinvent.chemistry.library_design.dtos import ReactionDTO, ReactionOutcomeDTO


class FragmentReactions:
    def __init__(self):
        self._bond_mapper = BondMapper()

    def create_reactions_from_smarts(self, smarts: List[str]) -> List[ChemicalReaction]:
        reactions = [AllChem.ReactionFromSmarts(smirks) for smirks in smarts]
        return reactions

    def create_reaction_from_smirk(self, smirks: str) -> ReactionDTO:
        reaction = ReactionDTO(smirks, AllChem.ReactionFromSmarts(smirks))
        return reaction

    def create_reactions_from_smirks(self, smirks: List[str]) -> List[ReactionDTO]:
        reactions = [self.create_reaction_from_smirk(smirk) for smirk in smirks]
        return reactions

    def slice_molecule_to_fragments(
        self, molecule: Mol, reaction_dtos: List[ReactionDTO]
    ) -> List[Tuple[Mol]]:
        """
        This method applies a list of chemical reactions on a molecule and
        decomposes the input molecule to complementary fragments.
        :param molecule:
        :param reaction_dtos:
        :return: Different slicing combinations are returned.
        """
        list_of_outcomes = self.apply_reactions_on_molecule(molecule, reaction_dtos)
        all_outcomes = []

        for outcome in list_of_outcomes:
            all_outcomes.extend(outcome.reaction_outcomes)
        # TODO: the overall data processing is extremely slow. consider reducing redundancy here.
        return all_outcomes

    def apply_reactions_on_molecule(
        self, molecule: Mol, reaction_dtos: List[ReactionDTO]
    ) -> List[ReactionOutcomeDTO]:
        """Build list of possible splits of a molecule given multiple reactions."""
        list_of_outcomes = []
        for reaction_dto in reaction_dtos:
            outcome_dto = self.apply_reaction_on_molecule(molecule, reaction_dto)
            purged_outcome_dto = self._filter_pairs_with_no_ring_count_change(outcome_dto)
            list_of_outcomes.append(purged_outcome_dto)
        return list_of_outcomes

    def apply_reaction_on_molecule(
        self, molecule: Mol, reaction_dto: ReactionDTO
    ) -> ReactionOutcomeDTO:
        """Build list of possible splits of a molecule given a single reaction."""
        molecule = conversions.copy_mol(molecule)
        outcomes = reaction_dto.chemical_reaction.RunReactant(molecule, 0)
        outcome_dto = ReactionOutcomeDTO(reaction_dto.reaction_smarts, list(outcomes), molecule)
        return outcome_dto

    def _filter_pairs_with_no_ring_count_change(
        self, outcome_dto: ReactionOutcomeDTO
    ) -> ReactionOutcomeDTO:
        molecule_rings = RingCount(outcome_dto.targeted_molecule)
        acceptable_pairs = []
        for pair in outcome_dto.reaction_outcomes:
            if not self._detect_ring_break(molecule_rings, pair) and len(pair) == 2:
                acceptable_pairs.append(pair)
        outcome_dto.reaction_outcomes = acceptable_pairs
        return outcome_dto

    def _detect_ring_break(self, molecule_ring_count: int, pair: Tuple[Mol]) -> bool:
        reagent_rings = 0
        for reagent in pair:
            reagent_smiles = conversions.mol_to_smiles(reagent)
            reagent_mol = conversions.smile_to_mol(reagent_smiles)
            try:
                reagent_rings = reagent_rings + RingCount(reagent_mol)
            except:
                return True
        return molecule_ring_count != reagent_rings
