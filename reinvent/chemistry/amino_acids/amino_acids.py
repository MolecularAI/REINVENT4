import logging
from collections import Counter
from typing import List

from reinvent.chemistry import tokens

logger = logging.getLogger(__name__)


def construct_amino_acids_fragments(fillers: List[str], masked_inputs: List[str],
                                    add_O=True, remove_cyclization_numbers=True) -> List[str]:
    """
    Construct fragmented amino acids.
    :param fillers: a list of fillers,
        e.g. ["N[C@@H](CC(=O)O)C(=O)|N[C@@H](C)C(=O)", "N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)"]
    :param masked_inputs: a list of masked input sequence,
        e.g. ["N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O", "N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O"]
    :param add_O: add O in the end of each fragment to form animo acid
    :param remove_cyclization_numbers: remove cyclization numbers in amino acid
    :return: a list of sequences where each sequence consists of fragmented amino acids
    """
    results = []
    for output, input in zip(fillers, masked_inputs):
        fragments = input
        # Put filler in the masked position
        for replacement in output.split(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN):
            fragments = fragments.replace(tokens.PEPINVENT_MASK_TOKEN, replacement, 1)
        results.append(fragments)
    if add_O:
        logger.debug("Adding O in the end of each fragment to form amino acid")
        results = add_O_to_endof_fragment_amino_acids(results)
    if remove_cyclization_numbers:
        logger.debug(("Removing cyclization numbers in amino acid"))
        results = remove_cyclization(results)

    return results


def add_O_to_endof_fragment_amino_acids(sequences: List[str]) -> List[str]:
    """
    Add O to the end of fragment amino acid if not ends with O
    :param sequences: a list of sequences, e.g. [
        "N[C@@H](C)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](C)C(=O)|NCC(=O)O",
        "N[C@@H](C)C(=O)|N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)|NCC(=O)O"]
    :return: a list of sequences where each sequence consists of separated amino acids with O in the end, e.g. [
        "N[C@@H](C)C(=O)O|N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](C)C(=O)O|NCC(=O)O",
        "N[C@@H](C)C(=O)O|N[C@@H]([C@H](CC)C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](Cc1c[nH]cn1)C(=O)O|NCC(=O)O"]
    """
    return [
        tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN.join(
            [aa if aa.endswith("O") else aa + "O" for aa in sequence.split(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN)])
        for sequence in sequences
    ]


def remove_cyclization(sequences: List[str]) -> List[str]:
    """
    Remove cyclization numbers from sequence
    :param sequences: list of sequence
    :return: list of sequence without cyclization numbers
    """
    sequences_without_cyclization = []
    for sequence in sequences:
        amino_acids = sequence.split(tokens.ATTACHMENT_SEPARATOR_TOKEN)
        cleaned_aas = []
        for aa in amino_acids:
            digit_counts = Counter(filter(str.isdigit, aa))
            # digits with odd counts indicating ring cyclization
            cleaned_aas.append(''.join(c for c in aa if not c.isdigit() or digit_counts[c] % 2 == 0))
        sequences_without_cyclization.append(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN.join(cleaned_aas))
    return sequences_without_cyclization
