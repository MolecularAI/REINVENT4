"""Simple parser routines for TOML and JSON

FIXME: about everything
"""

__all__ = ["read_smiles_csv_file", "read_config", "write_json"]
import sys
import io
from pathlib import Path
import csv
import json
import yaml
from typing import List, Tuple, Union, Optional, Callable

import tomli

from rdkit import Chem

smiles_func = Callable[[str], str]
FMT_CONVERT = {"toml": tomli, "json": json, "yaml": yaml}
INPUT_FORMAT_CHOICES = tuple(FMT_CONVERT.keys())


def monkey_patch_yaml_load(fct):
    def load(filehandle, loader=yaml.SafeLoader) -> dict:
        """Monkey patch for PyYAML's load

        yaml.load requires a loader or yaml.safe_load with a default

        :param filehandle: the filehandle to read the YAML from
        :returns: the parsed dictionary
        """

        return fct(filehandle, loader)

    return load


def yaml_loads(s) -> dict:
    """loads() implementation for PyWAML

    PyWAML does not have loading from string

    :param s: the string to load
    :returns: the parsed dictionary
    """

    fh = io.StringIO(s)
    data = yaml.safe_load(fh)

    return data


# only read first YAML document
yaml.load = monkey_patch_yaml_load(yaml.load)
yaml.loads = yaml_loads


def has_multiple_attachment_points_to_same_atom(smiles) -> bool:
    """Check a molecule for multiple attachment points on one atom

    An attachment point is a dummy atoom ("[*]")

    :param smiles: the SMILES string
    :returns: True if multiple attachment points exist, False otherwise
    """

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


def read_config(filename: Optional[Path], fmt: str) -> dict:
    """Read a config file in TOML, JON or (Py)YAML (safe load) format.

    :param filename: name of input file to be parsed as TOML, if None read from stdin
    :param fmt: name of the format of the configuration
    :returns: parsed dictionary
    """

    pkg = FMT_CONVERT[fmt]
    
    if isinstance(filename, (str, Path)):
        with open(filename, "rb") as tf:
            config = pkg.load(tf)
    else:
        config_str = "\n".join(sys.stdin.readlines())
        config = pkg.loads(config_str)
    
    return config


def write_json(data: str, filename: str) -> None:
    """Write data into a JSON file

    :param data: data in a format JSON accepts
    :param filename: output filename
    """

    with open(filename, "w") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)
