"""Test for ExternalProcess component with empty metadata bug"""

import pytest
import json
import tempfile
import os
import sys
import numpy as np
import numpy.testing as npt

from reinvent_plugins.components.comp_external_process import ExternalProcess, Parameters


def create_test_script(script_content: str) -> str:
    """Helper to create a temporary Python script for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        return f.name


@pytest.fixture
def cleanup_scripts():
    """Fixture to clean up temporary scripts after tests"""
    scripts = []
    yield scripts
    for script in scripts:
        try:
            os.unlink(script)
        except:
            pass


def test_externalprocess_empty_metadata(cleanup_scripts):
    """Test ExternalProcess with empty metadata dict (the bug scenario)"""
    
    # Create a script that returns scores but NO extra metadata
    script = create_test_script("""
import sys
import json

# Read SMILES from stdin
smiles = sys.stdin.read().strip().split('\\n')

# Return scores with empty metadata (no extra keys besides 'predictions')
data = {
    "version": 1,
    "payload": {
        "predictions": [0.5, 0.6, 0.7]
    }
}

print(json.dumps(data))
""")
    cleanup_scripts.append(script)
    
    # Setup component
    params = Parameters(
        executable=[sys.executable],
        args=[script],
        property=["predictions"]
    )
    component = ExternalProcess(params)
    
    # Call with test SMILES
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    results = component(smiles)
    
    # Verify scores are correct
    npt.assert_array_equal(results.scores[0], [0.5, 0.6, 0.7])
    
    # Verify metadata is empty dict (not None)
    assert results.metadata == {}, f"Expected empty dict, got {results.metadata}"
    
    # This is where the bug manifests - when creating SmilesAssociatedComponentResults
    from reinvent_plugins.components.component_results import SmilesAssociatedComponentResults
    
    # This should NOT raise an error
    smiles_results = SmilesAssociatedComponentResults(
        component_results=results,
        smiles=smiles
    )
    
    # Verify all SMILES are present in the data dict
    assert len(smiles_results.data) == 3, f"Expected 3 SMILES, got {len(smiles_results.data)}"
    for smi in smiles:
        assert smi in smiles_results.data, f"SMILES '{smi}' missing from data"
    
    # Verify we can fetch scores
    fetched_scores = smiles_results.fetch_scores(smiles=smiles, transpose=False)
    assert len(fetched_scores) == 3
    assert fetched_scores[0] == (0.5,)
    assert fetched_scores[1] == (0.6,)
    assert fetched_scores[2] == (0.7,)


def test_externalprocess_with_metadata(cleanup_scripts):
    """Test ExternalProcess with populated metadata (should work before and after fix)"""
    
    # Create a script that returns scores AND metadata
    script = create_test_script("""
import sys
import json

smiles = sys.stdin.read().strip().split('\\n')

data = {
    "version": 1,
    "payload": {
        "predictions": [0.5, 0.6, 0.7],
        "confidence": [0.95, 0.87, 0.92],
        "model": ["A", "A", "B"]
    }
}

print(json.dumps(data))
""")
    cleanup_scripts.append(script)
    
    params = Parameters(
        executable=[sys.executable],
        args=[script],
        property=["predictions"]
    )
    component = ExternalProcess(params)
    
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    results = component(smiles)
    
    # Verify scores
    npt.assert_array_equal(results.scores[0], [0.5, 0.6, 0.7])
    
    # Verify metadata is extracted (everything except 'predictions')
    assert "confidence" in results.metadata
    assert "model" in results.metadata
    assert "predictions" not in results.metadata
    
    # Create SmilesAssociatedComponentResults
    from reinvent_plugins.components.component_results import SmilesAssociatedComponentResults
    
    smiles_results = SmilesAssociatedComponentResults(
        component_results=results,
        smiles=smiles
    )
    
    # Verify all SMILES are present
    assert len(smiles_results.data) == 3
    
    # Verify metadata is correctly associated with each SMILES
    assert smiles_results["CCO"].metadata == {"confidence": 0.95, "model": "A"}
    assert smiles_results["c1ccccc1"].metadata == {"confidence": 0.87, "model": "A"}
    assert smiles_results["CC(=O)O"].metadata == {"confidence": 0.92, "model": "B"}


def test_externalprocess_multiple_endpoints_empty_metadata(cleanup_scripts):
    """Test ExternalProcess with multiple endpoints and empty metadata"""
    
    script = create_test_script("""
import sys
import json

smiles = sys.stdin.read().strip().split('\\n')

data = {
    "version": 1,
    "payload": {
        "affinity": [0.5, 0.6],
        "selectivity": [0.8, 0.9]
    }
}

print(json.dumps(data))
""")
    cleanup_scripts.append(script)
    
    params = Parameters(
        executable=[sys.executable],
        args=[script],
        property=["affinity", "selectivity"]  # Two endpoints
    )
    component = ExternalProcess(params)
    
    smiles = ["CCO", "c1ccccc1"]
    results = component(smiles)
    
    # Verify two score arrays
    assert len(results.scores) == 2
    npt.assert_array_equal(results.scores[0], [0.5, 0.6])
    npt.assert_array_equal(results.scores[1], [0.8, 0.9])
    
    # Metadata should be empty
    assert results.metadata == {}
    
    # Create SmilesAssociatedComponentResults
    from reinvent_plugins.components.component_results import SmilesAssociatedComponentResults
    
    smiles_results = SmilesAssociatedComponentResults(
        component_results=results,
        smiles=smiles
    )
    
    # Verify all SMILES are present
    assert len(smiles_results.data) == 2
    
    # Verify scores for both endpoints
    assert smiles_results["CCO"].score == (0.5, 0.8)
    assert smiles_results["c1ccccc1"].score == (0.6, 0.9)
    
    # Verify metadata is empty for each
    assert smiles_results["CCO"].metadata == {}
    assert smiles_results["c1ccccc1"].metadata == {}


@pytest.mark.parametrize(
    "payload_data, property_names, expected_scores, expected_metadata_keys",
    [
        # Empty metadata case (the bug)
        (
            {"predictions": [0.5, 0.6]},
            ["predictions"],
            [[0.5, 0.6]],
            []
        ),
        # With metadata
        (
            {"predictions": [0.5, 0.6], "confidence": [0.9, 0.8]},
            ["predictions"],
            [[0.5, 0.6]],
            ["confidence"]
        ),
        # Multiple endpoints, no metadata
        (
            {"affinity": [0.5, 0.6], "selectivity": [0.8, 0.9]},
            ["affinity", "selectivity"],
            [[0.5, 0.6], [0.8, 0.9]],
            []
        ),
        # Multiple endpoints with metadata
        (
            {"affinity": [0.5, 0.6], "selectivity": [0.8, 0.9], "source": ["A", "B"]},
            ["affinity", "selectivity"],
            [[0.5, 0.6], [0.8, 0.9]],
            ["source"]
        ),
    ],
)
def test_externalprocess_metadata_variations(
    cleanup_scripts, payload_data, property_names, expected_scores, expected_metadata_keys
):
    """Parametrized test for various metadata scenarios"""
    
    # Create script that returns the specified payload
    script_content = f"""
import sys
import json

smiles = sys.stdin.read().strip().split('\\n')

data = {{
    "version": 1,
    "payload": {json.dumps(payload_data)}
}}

print(json.dumps(data))
"""
    script = create_test_script(script_content)
    cleanup_scripts.append(script)
    
    params = Parameters(
        executable=[sys.executable],
        args=[script],
        property=property_names
    )
    component = ExternalProcess(params)
    
    smiles = ["CCO", "c1ccccc1"]
    results = component(smiles)
    
    # Verify scores
    assert len(results.scores) == len(expected_scores)
    for i, expected in enumerate(expected_scores):
        npt.assert_array_equal(results.scores[i], expected)
    
    # Verify metadata keys
    assert set(results.metadata.keys()) == set(expected_metadata_keys)
    
    # Create SmilesAssociatedComponentResults - this is where the bug occurs
    from reinvent_plugins.components.component_results import SmilesAssociatedComponentResults
    
    smiles_results = SmilesAssociatedComponentResults(
        component_results=results,
        smiles=smiles
    )
    
    # Verify all SMILES are present (this fails with the bug)
    assert len(smiles_results.data) == len(smiles), \
        f"Expected {len(smiles)} SMILES in data, got {len(smiles_results.data)}"
    
    for smi in smiles:
        assert smi in smiles_results.data, f"SMILES '{smi}' missing from data dict"