"""Simple parser routines for TOML and JSON

FIXME: about everything
"""

__all__ = ["read_smiles_csv_file", "read_toml", "read_json", "write_json"]
import sys
import csv
import json
from typing import List, Tuple, Union, Optional, Callable

import tomli

from rdkit import Chem

smiles_func = Callable[[str], str]


def has_multiple_attachment_points_to_same_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise RuntimeError(f"Error: Input {smiles} is not a valid molecule")

    seen = set()

    for atom in mol.GetAtoms():
        if atom.HasProp("dummyLabel"):
            neighbours = atom.GetNeighbors()

            if len(neighbours) > 1:
                raise RuntimeError("Error: dummy atom is not terminal")

            idx = neighbours[0].GetIdx()

            if idx in seen:
                return True

            seen.add(idx)

    return False


def read_smiles_csv_file(
    filename: str,
    columns: Union[int, slice],
    delimiter: str = "\t",
    header: bool = False,
    actions: List[smiles_func] = None,
    remove_duplicates: bool = False,
) -> Union[List[str], List[Tuple]]:
    """Read a SMILES column from a CSV file

    FIXME: needs to be made more robust

    :param filename: name of the CSV file
    :param columns: what number of the column to extract
    :param delimiter: column delimiter, must be a single character
    :param header: whether a header is present
    :param actions: a list of callables that act on each SMILES (only Reinvent
                    and Mol2Mol)
    :param remove_duplicates: whether to remove duplicates
    :returns: a list of SMILES or a list of a tuple of SMILES
    """

    smilies = []
    frontier = set()

    with open(filename, "r") as csvfile:
        if header:
            csvfile.readline()

        reader = csv.reader(csvfile, delimiter=delimiter)

        for row in reader:
            stripped_row = "".join(row).strip()

            if not stripped_row or stripped_row.startswith("#"):
                continue

            if isinstance(columns, int):
                smiles = row[columns].strip()
                orig_smiles = smiles

                if actions:
                    for action in actions:
                        if callable(action) and smiles:
                            smiles = action(orig_smiles)

                if not smiles:
                    continue

                # lib/linkinvent
                if "." in smiles:  # assume this is the standard SMILES fragment separator
                    smiles = smiles.replace(".", "|")

            else:
                smiles = tuple(smiles.strip() for smiles in row[columns])
                tmp_smiles = smiles

                # FIXME: hard input check for libinvent / linkinvent
                #        for unsupported scaffolds containing multiple
                #        attachment points to the same atoms.
                # libinvent
                new_smiles = smiles[1]

                if "." in new_smiles:  # assume this is the standard SMILES fragment separator
                    new_smiles = new_smiles.replace(".", "|")

                if "|" in new_smiles:
                    if has_multiple_attachment_points_to_same_atom(smiles[0]):
                        raise ValueError(
                            f"Not supported: Smiles {new_smiles} contains multiple attachment points for the same atom"
                        )

                    tmp_smiles = (smiles[0], new_smiles)

                # linkinvent
                new_smiles = smiles[0]

                if "." in new_smiles:  # assume this is the standard SMILES fragment separator
                    new_smiles = new_smiles.replace(".", "|")

                if "|" in new_smiles:
                    if has_multiple_attachment_points_to_same_atom(smiles[1]):
                        raise ValueError(
                            f"Not supported: Smiles {new_smiles} contains multiple attachment points for the same atom"
                        )

                    tmp_smiles = (new_smiles, smiles[1])

                smiles = tmp_smiles

            # SMILES transformation may fail
            # FIXME: needs sensible way to report this back to the user
            if smiles:
                if (not remove_duplicates) or (not smiles in frontier):
                    smilies.append(smiles)
                    frontier.add(smiles)

    return smilies


def read_toml(filename: Optional[str]) -> dict:
    """Read a TOML file.

    :param filename: name of input file to be parsed as TOML, if None read from stdin
    """

    if isinstance(filename, str):
        with open(filename, "rb") as tf:
            config = tomli.load(tf)
    else:
        config_str = "\n".join(sys.stdin.readlines())
        config = tomli.loads(config_str)

    return config


def read_json(filename: Optional[str]) -> dict:
    """Read JSON file.

    :param filename: name of input file to be parsed as JSON, if None read from stdin
    """

    if isinstance(filename, str):
        with open(filename, "rb") as jf:
            config = json.load(jf)
    else:
        config_str = "\n".join(sys.stdin.readlines())
        config = json.loads(config_str)

    return config


def write_json(data: str, filename: str) -> None:
    """Write data into a JSON file

    :param data: data in a format JSON accepts
    :param filename: output filename
    """
    with open(filename, "w") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)
