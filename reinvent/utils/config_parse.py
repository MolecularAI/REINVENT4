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

from reinvent.datapipeline.filters.regex import SMILES_TOKENS_REGEX


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
    allowed_tokens: tuple[set],
    delimiter: str = "\t",
    header: bool = False,
    actions: List[smiles_func] = None,
    remove_duplicates: bool = False,
) -> Union[List[str], List[Tuple]]:
    """Read a SMILES column from a CSV file

    FIXME: needs to be made more robust

    :param filename: name of the CSV file
    :param columns: what number of the column to extract (TL reads 2, RL/sampling 1)
    :param allowed_tokens: allowed tokens for the model
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

            if isinstance(columns, int):  # RL, sampling, TL (Reinvent, Mol2Mol)
                smiles = row[columns].strip()
                orig_smiles = smiles

                if actions:
                    for action in actions:
                        if callable(action) and smiles:
                            smiles = action(orig_smiles)

                if not smiles:
                    continue

                # Linkinvent "warheads" (R-groups)
                smiles = smiles.replace(".", "|")
                validate_tokens(smiles, allowed_tokens)
            else:  # TL (Lib/Linkinvent)
                smiles_pair = [smiles.strip() for smiles in row[columns]]

                # FIXME: hard input check for libinvent / linkinvent
                #        for unsupported scaffolds containing multiple
                #        attachment points to the same atoms.

                # Libinvent: scaffold/linker, R-groups
                check_separator(smiles_pair, 1, 0)

                # Linkinvent: R-groups, linker/scaffold
                check_separator(smiles_pair, 0, 1)

                smiles = smiles_pair
                validate_tokens(smiles, allowed_tokens, True)

            if smiles:  # SMILES transformation may fail
                if isinstance(smiles, list):
                    smiles = tuple(smiles)

                if (not remove_duplicates) or (not smiles in frontier):
                    smilies.append(smiles)
                    frontier.add(smiles)

    return smilies


def validate_tokens(
    smiles: str | list, allowed_tokens: tuple[set], TL_special: bool = False
) -> set:
    """Validate the SMILES against supported tokens

    FIXME: may have to check standardized SMILES e.g. if input
           contains "[F]", RDkit would remove the brackets

    :param allowed_tokens: allowed tokens for the model
    :param columns: what number of the column to extract (TL reads 2, RL/sampling)
    :param smiles: the SMILES string
    :returns: invalid tokens
    """

    if allowed_tokens[1] or TL_special:  # Lib/Linkinvent
        invalid_tokens0 = find_invalid_tokens(smiles[0], allowed_tokens[0])
        invalid_tokens_collected = set()
        invalid_tokens1 = set()

        if allowed_tokens[1]:  # Transformer models
            invalid_tokens1 = find_invalid_tokens(smiles[1], allowed_tokens[1])

        for invalid_tokens in invalid_tokens0, invalid_tokens1:
            invalid_tokens_collected.update(invalid_tokens)
    else:
        invalid_tokens_collected = find_invalid_tokens(smiles, allowed_tokens[0])

    if invalid_tokens_collected:
        raise ValueError(
            f"Tokens {invalid_tokens_collected} in {smiles} are not supported by the model\n"
            f"Allowed tokens are: {allowed_tokens}"
        )


def find_invalid_tokens(smiles: str, allowed_tokens: set) -> set:
    """Collect invalid tokens

    :param smiles: the SMILES string
    """

    tokens = set(SMILES_TOKENS_REGEX.findall(smiles))
    tokens = {s for s in tokens if "*" not in s}  # Lib/Linkinvent input
    return tokens - allowed_tokens


def check_separator(smiles_pair: list[str], col: int, other_col: int) -> None:
    """Check if column for separator and if it has multiple attachment points

    This function modifies the smiles_pair if the fragment separator is "."

    :param smiles_pair: SMILES pair
    :param col: column with fragments
    :param other_col: column to check
    """

    smiles_pair[col] = smiles_pair[col].replace(".", "|")

    if "|" in smiles_pair[col]:
        if has_multiple_attachment_points_to_same_atom(smiles_pair[other_col]):
            raise ValueError(
                f"Not supported: Smiles {smiles_pair[col]} contains multiple attachment "
                "points for the same atom"
            )


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
