from typing import List, Tuple

import pandas as pd
from reinvent.chemistry.library_design.bond_maker import BondMaker

from reinvent.chemistry.conversions import Conversions

from reinvent.chemistry.library_design.attachment_points import AttachmentPoints
from reinvent.chemistry.library_design.enums import ScaffoldMemoryFieldsEnum
from reinvent.chemistry.library_design.reaction_definitions.blocks_for_compound_dto import (
    BuildingBlocksForCompoundDTO,
)
from reinvent.chemistry.library_design.reaction_definitions.building_block_pair_dto import (
    BuildingBlockPairDTO,
)
from reinvent.chemistry.library_design.reaction_definitions.leaving_groups_dto import (
    LeavingGroupsDTO,
)
from reinvent.chemistry.library_design.reaction_definitions.standard_definitions import (
    StandardDefinitions,
)


class BuildingBlocks:
    def __init__(self, reaction_definition_file: str):
        self._reactions_library = StandardDefinitions(reaction_definition_file)
        self._attachments = AttachmentPoints()
        self._scaffold_memory_fields = ScaffoldMemoryFieldsEnum()
        self._conversions = Conversions()
        self._bond_maker = BondMaker()

    def create(
        self, reaction_name: str, position: int, dataframe: pd.DataFrame
    ) -> List[BuildingBlocksForCompoundDTO]:
        leaving_group_pairs = self._reactions_library.get_leaving_group_pairs(reaction_name)
        compound_fragments = {
            compound: fragments
            for compound, fragments in zip(
                dataframe[self._scaffold_memory_fields.SMILES],
                dataframe[self._scaffold_memory_fields.SCAFFOLD],
            )
        }
        blocks_for_compounds = [
            self._create_building_blocks_for_compound(
                compound, leaving_group_pairs, compound_fragments[compound], reaction_name, position
            )
            for compound in compound_fragments.keys()
        ]
        return blocks_for_compounds

    def _create_building_blocks_for_compound(
        self,
        compound: str,
        leaving_group_pairs: List[LeavingGroupsDTO],
        scaffold_decorations: str,
        reaction_name: str,
        position: int,
    ) -> BuildingBlocksForCompoundDTO:
        numbered_scaffold, numbered_decorations = self._separate_scaffold_and_decorations(
            scaffold_decorations
        )
        blocks = self._create_building_blocks(
            leaving_group_pairs, numbered_scaffold, position, numbered_decorations[position]
        )
        compound_blocks = BuildingBlocksForCompoundDTO(compound, reaction_name, position, blocks)
        return compound_blocks

    def _separate_scaffold_and_decorations(self, scaffold_decorations) -> Tuple[str, List[str]]:
        scaffold, *decorations = scaffold_decorations.split("|")
        scaffold = self._attachments.add_attachment_point_numbers(scaffold, False)
        decorations = [
            self._attachments.add_attachment_point_numbers(decoration, False)
            for decoration in decorations
        ]
        return scaffold, decorations

    def _create_building_blocks(
        self,
        leaving_group_pairs: List[LeavingGroupsDTO],
        scaffold: str,
        attachment_position: int,
        decoration: str,
    ) -> List[BuildingBlockPairDTO]:
        building_blocks = [
            self._create_building_block_pair(
                scaffold, decoration, attachment_position, leaving_group
            )
            for leaving_group in leaving_group_pairs
        ]
        return building_blocks

    def _create_building_block_pair(
        self,
        scaffold: str,
        decoration: str,
        attachment_position: int,
        leaving_group: LeavingGroupsDTO,
    ) -> BuildingBlockPairDTO:
        scaffold_group = self._attachments.add_first_attachment_point_number(
            leaving_group.leaving_group_scaffold, attachment_position
        )
        scaffold_group_mol = self._conversions.smile_to_mol(scaffold_group)
        scaffold_mol = self._conversions.smile_to_mol(scaffold)
        scaffold_block_mol = self._bond_maker.join_molecule_fragments(
            scaffold_mol, scaffold_group_mol
        )
        scaffold_block = self._conversions.mol_to_smiles(scaffold_block_mol)

        decoration_group = self._attachments.add_first_attachment_point_number(
            leaving_group.leaving_group_decoration, 0
        )
        decoration_group_mol = self._conversions.smile_to_mol(decoration_group)
        decoration_mol = self._conversions.smile_to_mol(decoration)
        decoration_block_mol = self._bond_maker.join_molecule_fragments(
            decoration_mol, decoration_group_mol
        )
        decoration_block = self._conversions.mol_to_smiles(decoration_block_mol)

        building_block = BuildingBlockPairDTO(scaffold_block, decoration_block)
        return building_block
