import pytest
from reinvent.chemistry.amino_acids.amino_acids import (
    construct_amino_acids_fragments,
    add_O_to_endof_fragment_amino_acids,
    remove_cyclization,
)


@pytest.mark.parametrize(
    "sequences, expected_output",
    [
        (
            [
                "N[C@@H](C)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](C)C(=O)|NCC(=O)O",
                "N[C@@H](C)C(=O)|N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)|NCC(=O)O",
            ],
            [
                "N[C@@H](C)C(=O)O|N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](C)C(=O)O|NCC(=O)O",
                "N[C@@H](C)C(=O)O|N[C@@H]([C@H](CC)C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](Cc1c[nH]cn1)C(=O)O|NCC(=O)O",
            ],
        )
    ],
)
def test_add_O_to_endof_fragment_amino_acids(sequences, expected_output):
    result = add_O_to_endof_fragment_amino_acids(sequences)
    assert result == expected_output


@pytest.mark.parametrize(
    "sequences, expected_output",
    [
        (
            [
                "N[C@@H](C)C(=O)O|N[C@@H](CS1)C(=O)O|N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS1)C(=O)O",
                "N[C@@H](C)C(=O)O|N[C@@H]([C@H](CC)C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](Cc1c[nH]cn1)C(=O)O|NCC(=O)O",
            ],
            [
                "N[C@@H](C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O",
                "N[C@@H](C)C(=O)O|N[C@@H]([C@H](CC)C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](Cc1c[nH]cn1)C(=O)O|NCC(=O)O",
            ],
        )
    ],
)
def test_remove_cyclization(sequences, expected_output):
    result = remove_cyclization(sequences)
    assert result == expected_output


@pytest.mark.parametrize(
    "fillers, masked_inputs, add_O, remove_cyclization_numbers, expected_output",
    [
        (
            [
                "N[C@@H](CC(=O)O)C(=O)|N[C@@H](C)C(=O)",
                "N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)",
            ],
            [
                "N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O",
                "N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O",
            ],
            True,
            True,
            [
                "N[C@@H](C)C(=O)O|N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](C)C(=O)O|NCC(=O)O",
                "N[C@@H](C)C(=O)O|N[C@@H]([C@H](CC)C)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](Cc1c[nH]cn1)C(=O)O|NCC(=O)O",
            ],
        ),
        (
            [
                "N[C@@H](CC(=O)O)C(=O)|N[C@@H](C)C(=O)",
                "N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)",
            ],
            [
                "N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O",
                "N[C@@H](C)C(=O)|?|N[C@@H](CS)C(=O)|?|NCC(=O)O",
            ],
            False,
            False,
            [
                "N[C@@H](C)C(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](C)C(=O)|NCC(=O)O",
                "N[C@@H](C)C(=O)|N[C@@H]([C@H](CC)C)C(=O)|N[C@@H](CS)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)|NCC(=O)O",
            ],
        ),
    ],
)
def test_construct_amino_acids_fragments(
    fillers, masked_inputs, add_O, remove_cyclization_numbers, expected_output
):
    result = construct_amino_acids_fragments(
        fillers, masked_inputs, add_O, remove_cyclization_numbers
    )
    assert result == expected_output
