import numpy.testing as npt
import pytest

from reinvent_plugins.components.comp_mmp import MMP, Parameters
from tests.test_data import CHEMBL4442703, CHEMBL4469449, CHEMBL4546342
from tests.test_data import CHEMBL4458766, CHEMBL4530872, CHEMBL4475647
from tests.test_data import CHEMBL4440201, CHEMBL4475970


@pytest.mark.integration
@pytest.mark.parametrize(
    "reference_smiles, expected_results",
    [
        ([[CHEMBL4440201]], ["MMP", "MMP", "MMP", "No MMP", "No MMP", "No MMP"]),
        (
            [[CHEMBL4440201, CHEMBL4475970]],
            ["MMP", "MMP", "MMP", "No MMP", "MMP", "MMP"],
        ),
    ],
)
def test_comp_mmp(reference_smiles, expected_results):
    query_smiles = [
        CHEMBL4442703,
        CHEMBL4469449,
        CHEMBL4546342,
        CHEMBL4458766,
        CHEMBL4530872,
        CHEMBL4475647,
    ]
    params = Parameters(reference_smiles=reference_smiles)
    component = MMP(params)
    results = component(query_smiles)
    npt.assert_array_equal(results.scores[0], expected_results)
