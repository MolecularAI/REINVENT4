"""
This file contains functions that collect or extract information from a tree,
functions that "reduce" or ("fold" or "accumulate" or "aggregate") a tree to some value,
e.g. a scalar value like the depth of the tree,
or a list with SMILES strings of all intermediates in the tree.

AiZynthFinder trees contain three types of nodes:
  1. Starting material: node["type"] == "mol" and "children" not in node
  2. Intermediates, including the product: node["type"] == "mol" and "children" in node
  3. Reactions: node["type"] == "reaction" and "children" in node
"""


def is_solved(node: dict) -> bool:
    """Return True if a tree was solved.

    AiZynthFinder return all trees,
    even the trees where it failed to solve the tree,
    and starting material is not in stock.
    """
    if node["type"] == "mol" and "children" not in node:  # Starting material
        return node["in_stock"]
    else:  # Intermediate or reaction
        return all(is_solved(c) for c in node["children"])


def depth(node):
    """Return a depth of the synthesis tree.

    Adds 1 for each reaction, and 0 for molecules, recursively.
    """

    if node["type"] == "reaction":  # reaction
        return 1 + max(depth(c) for c in node["children"])
    elif node["type"] == "mol" and "children" in node:  # intermediate
        return max(depth(c) for c in node["children"])
    else:  # starting material
        return 0


def startmat(node: dict) -> list:
    """Return a list of starting materials."""
    if node["type"] == "mol" and "children" not in node:  # Starting material
        return [node["smiles"]]
    else:  # Intermediate or reaction
        return sum([startmat(c) for c in node["children"]], [])


def reaction_classes(node: dict) -> list:
    """Return a list of reaction classes.

    Reactions are returned in depth-first order.
    """
    if node["type"] == "reaction":
        return [node["metadata"]["classification"]] + sum(
            [reaction_classes(c) for c in node["children"]], []
        )
    elif node["type"] == "mol" and "children" in node:
        return sum([reaction_classes(c) for c in node["children"]], [])
    else:
        return []


def leaves_from_routes(routes):
    intermediates = []
    for route in routes:
        intermediates.extend(route.leaves())
    return intermediates


def reaction_classes_with_depth(node: dict, depth=1) -> list:
    """Return a list of reaction classes with their depth (level) in the tree."""
    if node["type"] == "reaction":
        return [(node["metadata"]["classification"], depth)] + sum(
            [reaction_classes_with_depth(c, depth + 1) for c in node["children"]], []
        )
    elif node["type"] == "mol" and "children" in node:
        return sum([reaction_classes_with_depth(c, depth) for c in node["children"]], [])
    else:
        return []


def intermediates(node: dict) -> list:
    """Return a list of intermediates."""
    if node["type"] == "mol" and "children" not in node:  # startmat
        return []
    elif node["type"] == "mol" and "children" in node:  # intermediate
        return [node["smiles"]] + sum([intermediates(c) for c in node["children"]], [])
    else:  # reaction
        return sum([intermediates(c) for c in node["children"]], [])


def intermediates_with_depth(node: dict, depth=0) -> list:
    """Return a list of intermediates with their depth in the tree."""
    if node["type"] == "mol" and "children" not in node:  # startmat
        return []
    elif node["type"] == "mol" and "children" in node:  # intermediate
        return [(node["smiles"], depth)] + sum(
            [intermediates_with_depth(c, depth + 1) for c in node["children"]],
            [],
        )
    else:  # reaction
        return sum([intermediates_with_depth(c, depth) for c in node["children"]], [])


def trim(node: dict, intermediates: list):
    if node["type"] == "mol" and "children" not in node:
        return node
    elif node["type"] == "mol" and node["smiles"] in intermediates:
        # Return node without children, as if it was a starting material:
        node_copy = node.copy()
        node_copy.pop("children", None)  # remove children, if any
        return node_copy
    else:  # reaction or non-common intermediate
        node_copy = node.copy()
        new_children = [trim(c, intermediates) for c in node["children"]]
        node_copy["children"] = new_children
        return node_copy


def pretty_string(node: dict, indent="") -> str:
    """Pretty-print the tree to a string."""
    if node["type"] == "reaction":
        return f"{indent}{node['metadata']['classification']}\n" + "\n".join(
            [pretty_string(c, indent + "  ") for c in node["children"]]
        )
    elif node["type"] == "mol" and "children" in node:
        return f"{indent}{node['smiles']}\n" + "\n".join(
            [pretty_string(c, indent + "  ") for c in node["children"]]
        )
    else:
        return indent + node["smiles"]
