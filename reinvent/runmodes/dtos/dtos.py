"""Various data classes defining DTOs needed in RL sampling."""

from dataclasses import dataclass
from typing import List

import torch


# Reinvent
@dataclass
class SampledBatchDTO:
    sequences: torch.Tensor  # 2D: tokens per sequence, each token is integer
    smiles: List[str]
    nlls: torch.Tensor  # 1D: negative log likelihoods, floats


@dataclass
class UpdatedLikelihoodsDTO:
    agent_likelihood: torch.Tensor
    prior_likelihood: torch.Tensor
    augmented_likelihood: torch.Tensor
    loss: torch.Tensor
