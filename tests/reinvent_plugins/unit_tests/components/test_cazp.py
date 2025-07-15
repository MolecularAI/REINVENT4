import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt


from reinvent_plugins.components.cazp.aizynthfinder_config import ensure_custom_stock_is_inchikey
from reinvent_plugins.components.cazp.comp_cazp import (
    Cazp,
    Parameters,
)

from reinvent_plugins.components.cazp.endpoints import CazpEndpoint

from reinvent_plugins.components.cazp.tree_edit_distance import TED


@pytest.fixture
def cazp_params():
    params = Parameters()
    params.score_to_extract = ["cazp"]
    yield params


@pytest.fixture
def cazp_scoring_component_instance(cazp_params):
    return Cazp(cazp_params)


@pytest.fixture
def cazp_endpoint_instance():
    return CazpEndpoint()


def test_cazp_init_single_endpoint(cazp_params):
    cazp_params.number_of_steps = [10, 20]
    with pytest.raises(ValueError):
        cazp = Cazp(cazp_params)


def test_cazp_init_single_endpoint_explicit(cazp_params):
    cazp_params.score_to_extract = ["cazp"]
    cazp_params.number_of_steps = [1]
    cazp = Cazp(cazp_params)
    assert cazp.params.score_to_extract == ["cazp"]


def test_cazp_init_multiple_endpoints_correct_score_to_extract(cazp_params):
    cazp_params.score_to_extract = ["cazp", "route_distance"]
    cazp = Cazp(cazp_params)
    assert cazp.params.score_to_extract == ["cazp", "route_distance"]


def test_cazp_init_multiple_endpoints_no_score_to_extract(cazp_params):
    cazp_params.score_to_extract = None
    cazp_params.number_of_steps = [1, 2]
    with pytest.raises(ValueError):
        Cazp(cazp_params)


def test_cazp_init_multiple_endpoints_missing_score_to_extract(cazp_params):
    cazp_params.score_to_extract = ["route_distance"]
    cazp_params.number_of_steps = [1, 2]
    with pytest.raises(ValueError):
        Cazp(cazp_params)


def test_ensure_custom_stock_is_inchikey_v3(tmp_path):
    smi_content = "CCO\nCCC\n"
    smi_file = tmp_path / "test_stock.smi"
    with open(smi_file, "w") as f:
        f.write(smi_content)

    config = {"stock": {"files": {"stock1": str(smi_file)}}}

    ensure_custom_stock_is_inchikey(config, str(tmp_path))

    assert "stock1" in config["stock"]["files"]
    assert config["stock"]["files"]["stock1"].endswith(".csv")
    csv_file = config["stock"]["files"]["stock1"]
    df = pd.read_csv(csv_file)
    assert "smiles" in df.columns
    assert "inchi_key" in df.columns
    assert df["smiles"].tolist() == ["CCO", "CCC"]


def test_ensure_custom_stock_is_inchikey_v4(tmp_path):
    smi_content = "CCO\nCCC\n"
    smi_file1 = tmp_path / "test_stock.smi"
    smi_file2 = tmp_path / "test_stock2.smi"
    with open(smi_file1, "w") as f:
        f.write(smi_content)
    with open(smi_file2, "w") as f:
        f.write(smi_content)

    config = {"stock": {"stock1": str(smi_file1), "stock2": {"path": str(smi_file2)}}}

    ensure_custom_stock_is_inchikey(config, str(tmp_path))

    assert "stock1" in config["stock"]
    assert config["stock"]["stock1"].endswith(".csv")
    csv_file1 = config["stock"]["stock1"]
    df1 = pd.read_csv(csv_file1)
    assert "smiles" in df1.columns
    assert "inchi_key" in df1.columns
    assert df1["smiles"].tolist() == ["CCO", "CCC"]

    assert "stock2" in config["stock"]
    assert config["stock"]["stock2"]["path"].endswith(".csv")
    csv_file2 = config["stock"]["stock2"]["path"]
    df2 = pd.read_csv(csv_file2)
    assert "smiles" in df2.columns
    assert "inchi_key" in df2.columns
    assert df2["smiles"].tolist() == ["CCO", "CCC"]


def test_extract_endpoint_single_tree(cazp_endpoint_instance):
    smilies = ["CCO"]
    out = {
        "data": [
            {
                "target": "CCO",
                "trees": [
                    {
                        "scores": {
                            "stock availability": 0.8,
                            "number of reactions": 2,
                            "reaction class membership": 1,
                        }
                    }
                ],
            }
        ]
    }
    cazp_endpoint_instance.reaction_step_coefficient = 0.9
    scores = cazp_endpoint_instance.get_scores(smilies, out)
    npt.assert_almost_equal(scores, np.array([0.648]))


def test_extract_endpoint_multiple_trees(cazp_endpoint_instance):
    smilies = ["CCO"]
    out = {
        "data": [
            {
                "target": "CCO",
                "trees": [
                    {
                        "scores": {
                            "cazp": 0.8,
                            "number of reactions": 2,
                            "reaction class membership": 1,
                            "stock availability": 0.8,
                        }
                    },
                    {
                        "scores": {
                            "cazp": 0.9,
                            "reaction class membership": 1,
                            "number of reactions": 2,
                            "stock availability": 0.8,
                        }
                    },
                ],
            }
        ]
    }
    scores = cazp_endpoint_instance.get_scores(smilies, out)
    npt.assert_array_almost_equal(scores, np.array([0.648]))


def test_tree_edit_distance_one_undef():

    t1 = {
        "smiles": "",
        "type": "mol",
        "children": [
            {
                "type": "reaction",
                "metadata": {"classification": "1.2.3 SomeClass"},
                "children": [{"type": "mol", "smiles": ""}],
            }
        ],
    }
    t2 = {
        "smiles": "",
        "type": "mol",
        "children": [
            {
                "type": "reaction",
                "metadata": {"classification": "0.0 Undef"},
                "children": [{"type": "mol", "smiles": ""}],
            }
        ],
    }
    assert TED(t1, t2) == 1


def test_tree_edit_distance_both_undef():

    t1 = {
        "smiles": "",
        "type": "mol",
        "children": [
            {
                "type": "reaction",
                "metadata": {"classification": "0.0 Undef"},
                "children": [{"type": "mol", "smiles": ""}],
            }
        ],
    }
    t2 = {
        "smiles": "",
        "type": "mol",
        "children": [
            {
                "type": "reaction",
                "metadata": {"classification": "0.0 Undef"},
                "children": [{"type": "mol", "smiles": ""}],
            }
        ],
    }
    assert TED(t1, t2) == 1
