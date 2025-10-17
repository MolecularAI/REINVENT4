import pytest

import numpy as np

from reinvent.runmodes.RL.memories.identical_murcko_scaffold import IdenticalMurckoScaffold
from reinvent.runmodes.RL.memories.identical_murcko_scaffold_rnd import IdenticalMurckoScaffoldRND
from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState


@pytest.fixture(params=["Step", "Tanh", "Sigmoid", "Linear", "Erf"])
def diversity_filter(request):

    diversity_filter = IdenticalMurckoScaffold(
        bucket_size=2,
        minscore=0.4,
        minsimilarity=0.4,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
        device=None,
        prior_model_file_path=None,
        penalty_function=request.param,
        learning_rate=None,
    )

    yield diversity_filter

    pass  # no tear-down


@pytest.fixture(params=["Step", "Tanh", "Sigmoid", "Linear", "Erf"])
def diversity_filter_rnd(device, json_config, request):

    diversity_filter = IdenticalMurckoScaffoldRND(
        bucket_size=2,
        minscore=0.4,
        minsimilarity=0.4,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
        device=device,
        prior_model_file_path=json_config["PRIOR_PATH"],
        penalty_function=request.param,
        learning_rate=1e-4,
    )

    yield diversity_filter

    pass  # no tear-down


def test_filter(diversity_filter):

    scores = [0.64, 0.55, 0.897, 0.737]
    smilies = ["CCc1ccccc1", "CCC=O=O", "CCCCc1ccccc1", "CCc1cnccc1"]
    states = np.array(
        [SmilesState.VALID, SmilesState.INVALID, SmilesState.VALID, SmilesState.DUPLICATE]
    )

    # DF needs all valid SMILES including duplicates
    mask = np.where(
        (states == SmilesState.VALID) | (states == SmilesState.DUPLICATE),
        True,
        False,
    )

    diversity_filter.update_score(scores, smilies, mask, None)

    assert diversity_filter.scaffold_memory.max_size == 2
    assert diversity_filter.minscore == 0.4
    assert diversity_filter.minsimilarity == 0.4
    assert diversity_filter.penalty_multiplier == 0.5
    assert len(diversity_filter.scaffold_memory) == 2


def test_filter_rnd(diversity_filter_rnd):
    
    scores = [0.64, 0.55, 0.897, 0.737]
    smilies = ["CCc1ccccc1", "CCC=O=O", "CCCCc1ccccc1", "CCc1cnccc1"]
    states = np.array(
        [SmilesState.VALID, SmilesState.INVALID, SmilesState.VALID, SmilesState.DUPLICATE]
    )

    # DF needs all valid SMILES including duplicates
    mask = np.where(
        (states == SmilesState.VALID) | (states == SmilesState.DUPLICATE),
        True,
        False,
    )

    sampled = SampleBatch(None, smilies, scores)

    diversity_filter_rnd.update_score(scores, smilies, mask, sampled)

    assert diversity_filter_rnd.scaffold_memory.max_size == 2
    assert diversity_filter_rnd.minscore == 0.4
    assert diversity_filter_rnd.minsimilarity == 0.4
    assert diversity_filter_rnd.penalty_multiplier == 0.5
    assert len(diversity_filter_rnd.scaffold_memory) == 2
