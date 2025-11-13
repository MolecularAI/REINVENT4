import pytest
import numpy.testing as npt

from unittest.mock import mock_open, patch
from reinvent_plugins.components.database_precedence.comp_ringprecedence import (
    RingPrecedence,
    Parameters,
)


@pytest.mark.integration
def test_comp_ringprecedence():
    inputs = [
        "c1ccccc1Cc1ccccc1",
        "Brc1ccncc1",
        "BrC1C=CC=C1",
        "FC1CCC1",
        "CCCCC",
    ]  # in db, in db, outside of db/in generic, out of both, linear
    database_string = """
    {"rings": {"c1ccccc1": 0.98, "c1ccncc1": 2.91, "linear": 5},
    "generic_rings": {"C1CCCCC1": 0.79, "C1CCCC1": 1.72}
    } """
    # Create a temporary file
    mock_file = mock_open(read_data=database_string)

    with patch("builtins.open", mock_file):
        # test total, rings
        expected_results = [0.98 * 2, 2.91, 100, 100, 5]
        params = Parameters(
            nll_method=["total"], database_file=["/mocked/path"], make_generic=[False]
        )
        component = RingPrecedence(params)
        results = component(inputs)
        npt.assert_array_equal(results.scores[0], expected_results)

        # test max, rings
        expected_results = [0.98, 2.91, 100, 100, 5]
        params = Parameters(
            nll_method=["max"], database_file=["/mocked/path"], make_generic=[False]
        )
        component = RingPrecedence(params)
        results = component(inputs)
        npt.assert_array_equal(results.scores[0], expected_results)

        # test generic
        expected_results = [0.79, 0.79, 1.72, 100, 100]
        params = Parameters(nll_method=["max"], database_file=["/mocked/path"], make_generic=[True])
        component = RingPrecedence(params)
        results = component(inputs)
        npt.assert_array_equal(results.scores[0], expected_results)
        assert "highest_nll_ring" in results.metadata
        assert len(results.metadata["highest_nll_ring"]) == len(inputs)

