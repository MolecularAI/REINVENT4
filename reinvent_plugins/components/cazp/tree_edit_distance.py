from functools import lru_cache

import numpy as np
from apted import APTED, Config


@lru_cache
def get_reaction_classifications(classification: str) -> list[int]:
    """Splits AiZynthFinder matadata into numbers"""

    rc = classification.split(" ")[0]  # Take numeric classes, ignore text name.
    rcn = [int(i) for i in rc.split(".")]
    return rcn


class CustomConfig(Config):
    """Custom APTED config for CAZP"""

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        return 1

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        return 1

    def rename(self, node1, node2):
        if node1["type"] != node2["type"]:
            return 1

        if node1["type"] == "reaction":

            rc1n = get_reaction_classifications(node1["metadata"]["classification"])
            rc2n = get_reaction_classifications(node2["metadata"]["classification"])
            if len(rc1n) != len(rc2n):
                return 1
            if len(rc1n) == 0:  # No classification found.
                return 1
            if rc1n[0] == 0 or rc2n[0] == 0:  # Superclass is 0, Undefined.
                return 1

            diffs = ~np.equal(rc1n, rc2n)
            if len(diffs) < 3:
                return 1  # Not enough classes to compare, probably one is Undefined.

            if diffs[0]:  # Diff in first number.
                dist = 0.8
            elif diffs[1]:
                dist = 0.5
            elif diffs[2]:
                dist = 0
            else:  # No diff.
                dist = 0
            return dist

        if node1["type"] == "mol":
            return 0

    def children(self, node):
        """Get children of a tree"""
        return node.get("children", [])


def TED(tree1, tree2):
    """Tree edit distance between two trees"""

    apted = APTED(tree1, tree2, CustomConfig())
    return apted.compute_edit_distance()
