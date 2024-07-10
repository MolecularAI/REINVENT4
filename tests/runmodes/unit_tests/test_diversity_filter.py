import pytest

import numpy as np

from reinvent.runmodes.RL.memories.identical_murcko_scaffold import IdenticalMurckoScaffold
from reinvent.models.model_factory.sample_batch import SmilesState


@pytest.fixture
def diversity_filter():

    diversity_filter = IdenticalMurckoScaffold(
        bucket_size=2,
        minscore=0.4,
        minsimilarity=0.4,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
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

    diversity_filter.update_score(scores, smilies, mask)

    assert diversity_filter.scaffold_memory.max_size == 2
    assert diversity_filter.minscore == 0.4
    assert diversity_filter.minsimilarity == 0.4
    assert diversity_filter.penalty_multiplier == 0.5
    assert len(diversity_filter.scaffold_memory) == 2
