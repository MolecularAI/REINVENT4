from typing import List

import pandas as pd

from reinvent.chemistry.library_design.reaction_definitions.leaving_groups_dto import (
    LeavingGroupsDTO,
)


class StandardDefinitions:
    def __init__(self, definitions_path: str):
        # TODO: implement in config default path for reaction definitions
        self._definitions: pd.DataFrame = self.load_definitions(definitions_path)

    def load_definitions(self, definitions_path: str):
        """Reads a csv file with named reaction definitions and leaving groups"""
        columns = ["id", "name", "retro_reaction", "group_1", "group_2"]
        try:
            definitions = pd.read_csv(definitions_path, skipinitialspace=True, usecols=columns)
        except:
            raise FileExistsError(f"the specified path is missing {definitions_path}")
        return definitions

    def get_reaction_definition(self, name: str) -> str:
        """Returns a single retro-reaction definition SMIRKS"""
        result = self._definitions.query("name == @name")
        if len(result) > 0:
            return result["retro_reaction"].iloc[0]
        else:
            raise IOError(f"there are no definitions for reaction name: {name}")

    def get_leaving_group_pairs(self, name: str) -> List[LeavingGroupsDTO]:
        """Returns a list of leaving group pairs"""
        result = self._definitions.query("name == @name")

        if len(result) == 0:
            raise IOError(f"there are no definitions for reaction name: {name}")

        leaving_groups = [
            LeavingGroupsDTO(g1.replace("''", ""), g2.replace("''", ""))
            for g1, g2 in zip(result["group_1"], result["group_2"])
        ]
        permutated_leaving_groups = [
            LeavingGroupsDTO(g2.replace("''", ""), g1.replace("''", ""))
            for g1, g2 in zip(result["group_1"], result["group_2"])
        ]
        leaving_groups.extend(permutated_leaving_groups)
        return leaving_groups
