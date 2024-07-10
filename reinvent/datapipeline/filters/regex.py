"""Filter SMILES with regular expressions

Assumes correct SMILES syntax e.g. that elements other than the basic ones are
in brackets.
"""

from __future__ import annotations

__all__ = ["RegexFilter", "SMILES_TOKENS_REGEX"]
import re
from collections import Counter
from typing import Optional, TYPE_CHECKING

from . import elements

if TYPE_CHECKING:
    from ..validation import FilterSection

# adapted from SmilesPE-0.0.3
SMILES_TOKENS_REGEX = re.compile(
    r"(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%\d+|\d)"
)
ELEMENT = re.compile(r"[A-Za-z]+")
ISOTOPE = re.compile(r"\[\d+")
ISOTOPE_EXTRACT = re.compile("[A-Za-z@+-]+\d*")
STEREO = re.compile(r"\w+@{1,2}[H+-]?")
ATOM_MAP = re.compile(r"[^[].*:\d+")

# FIXME: RDKit may still create charged halogens
UNWANTED_TOKENS = re.compile(r"\[\d*(F|Cl|Br|I)(H|[+-]|\d|@)?.*]")

NON_BRACKET_ATOMS = {"B", "C", "N", "O", "S", "P", "F", "I"}


class RegexFilter:
    """Filter SMILES with the help of regular expressions"""

    def __init__(self, config: FilterSection):
        self.config = config
        self.elements = self.config.elements

        if not elements.valid_elements(self.elements):
            raise RuntimeError(f"Invalid elements in {self.elements}")

        self.discarded_tokens = Counter()
        self.token_count = 0

    def __call__(self, smiles: str) -> Optional[str]:
        """Process and filter SMILES

        May modify SMILES because of 3 and 4 digit ring numbers.

        :param smiles: input SMILES
        :returns: processed SMILES or None if SMILES is skipped
        """

        if not smiles:
            return None

        new_smiles = []
        heavy_atom_count = 0
        carbon_count = 0
        mol_wt = 0

        tokens = SMILES_TOKENS_REGEX.findall(smiles)

        for token in tokens:
            new_token = token
            self.token_count += 1

            # Discard unwanted tokens
            # FIXME: Make this configurable?
            if UNWANTED_TOKENS.search(token):
                return None

            if ISOTOPE.match(token):
                if self.config.keep_isotope_molecules:
                    extract = ISOTOPE_EXTRACT.search(token).group(0)
                    new_token = get_pattern(extract)
                else:
                    return None

            if (not self.config.keep_stereo) and (match := STEREO.search(new_token)):
                extract = match.group(0).replace("@", "")
                new_token = get_pattern(extract)

            if match := ATOM_MAP.search(new_token):
                m = match.group(0)
                idx = m.find(":")
                extract = m[:idx]
                new_token = get_pattern(extract)

            if match := ELEMENT.search(new_token):
                elem = match.group(0).rstrip("H").title()

                if elem not in self.elements:
                    self.discarded_tokens.update([token])
                    return None

                # FIXME: this may not capture all possibilities
                if elem.startswith("C") or elem.startswith("c") or elem.startswith("#6"):
                    carbon_count += 1

                if elem != "H":
                    heavy_atom_count += 1

                mol_wt += elements.PERIODIC_TABLE[elem]

            if heavy_atom_count > self.config.max_heavy_atoms:
                return None

            if mol_wt > self.config.max_mol_weight:
                return None

            new_smiles.append(new_token)

        if heavy_atom_count < self.config.min_heavy_atoms:
            return None

        if carbon_count < self.config.min_carbons:
            return None

        new_smiles = "".join(new_smiles)

        return new_smiles


def get_pattern(extract):
    if extract == "H":
        pattern = f"[{extract}]"  # FIXME: may need review
    elif extract in NON_BRACKET_ATOMS:
        pattern = extract
    else:
        pattern = f"[{extract}]"

    return pattern
