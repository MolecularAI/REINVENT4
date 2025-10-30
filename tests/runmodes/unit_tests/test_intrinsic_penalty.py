import pytest

import numpy as np

from reinvent.runmodes.RL.intrinsic_penalty.identical_murcko_scaffold_rnd import (
    IdenticalMurckoScaffoldRND,
)
from reinvent.models.model_factory.sample_batch import SmilesState, SampleBatch
from reinvent.runmodes.RL.validation import SectionIntrinsicPenalty
from tests.runmodes.integration_tests.sampling_tests.test_sampling import param


@pytest.fixture(params=["Step", "Tanh", "Sigmoid", "Linear", "Erf"])
def intrinsic_penalty(device, json_config, request):

    intrinsic_penalty = IdenticalMurckoScaffoldRND(
        penalty_function=request.param,
        bucket_size=2,
        minscore=0.4,
        device=device,
        prior_model_file_path=json_config["PRIOR_PATH"],
        learning_rate=1e-4,
        rdkit_smiles_flags={},
    )

    yield intrinsic_penalty

    pass  # no tear-down


def test_filter(intrinsic_penalty):
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

    intrinsic_penalty.update_score(scores, smilies, mask, sampled)

    assert intrinsic_penalty.scaffold_memory.max_size == 2
    assert intrinsic_penalty.minscore == 0.4
    assert len(intrinsic_penalty.scaffold_memory) == 2
