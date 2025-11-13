from functools import lru_cache

import numpy as np
from apted import APTED, Config

from reinvent_plugins.components.synthsense.reducers import reaction_classes


@lru_cache()
def nextmove_top2(classification: str) -> str:
    """Return 2 top-level classification parts in NextMove reaction classification

    NextMove uses three levels:
        1. Superclass
        2. Class
        3. Named reaction

    We keep only superclass and class.

    Example: "2.1.1 Amide Schotten-Baumann" -> "2.1"
    """

    top2 = classification.split(" ")[0].split(".")[0:2]
    return ".".join(top2)


def route_signature(node: dict) -> str:
    """Return a signature for a route

    The signature is a string with all reaction classes in the tree.
    """
    return ",".join([nextmove_top2(r) for r in reaction_classes(node)])


@lru_cache
def get_reaction_classifications(classification: str) -> list[int]:
    """Splits AiZynthFinder matadata into numbers"""

    rc = classification.split(" ")[0]  # Take numeric classes, ignore text name.
    rcn = [int(i) for i in rc.split(".")]
    return rcn


class CustomConfig(Config):
    """Custom APTED config for synthsense"""

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        return 4

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        return 4

    def rename(self, node1, node2):
        if node1["type"] != node2["type"]:
            return 4

        if node1["type"] == "reaction":

            rc1n = get_reaction_classifications(node1["metadata"]["classification"])
            rc2n = get_reaction_classifications(node2["metadata"]["classification"])
            if len(rc1n) != len(rc2n):
                return 1
            diffs = ~np.equal(rc1n, rc2n)
            if diffs[0]:  # Diff in first number.
                dist = 3.0
            elif len(diffs) > 1 and diffs[1]:
                dist = 2.0
            elif len(diffs) > 2 and diffs[2]:
                dist = 1.0
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
