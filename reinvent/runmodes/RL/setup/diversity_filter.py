from __future__ import annotations

import logging

import torch

from reinvent.runmodes.RL.memories import penalties
from reinvent.runmodes.RL.memories import intrinsic_rewards
from reinvent.runmodes.RL.memories import filters
from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
from reinvent.runmodes.RL.memories.intrinsic_reward_base import IntrinsicReward
from reinvent.runmodes.RL.memories.penalty_base import BasePenalty
from reinvent.runmodes.RL.validation import SectionDiversityFilter

logger = logging.getLogger(__name__)


def setup_diversity_filter(
    config: SectionDiversityFilter,
    device: torch.device,
    prior_model_file_path: str,
    rdkit_smiles_flags: dict,
):
    """Setup of the diversity filter

    Basic setup of the diversity filter memory.  The parameters are from a
    dict, so the keys (parameters) are hard-coded here.

    :param config: config parameter specific to the filter
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: the set up diversity filter
    """

    if config is None or not hasattr(config, "type"):
        return None

    diversity_filter: type[DiversityFilter] = getattr(filters, config.type)

    logger.info(f"Using diversity filter {config.type}")

    # Setup Penalty Function
    penalty_class: type[BasePenalty] = getattr(
        penalties, f"{config.penalty_function}Penalty"
    )

    penalty = penalty_class(config.bucket_size)  # Ensure the penalty class is loaded

    logger.info(f"Using penalty function: {penalty.__class__.__name__}")

    intrinsic_reward: IntrinsicReward | None = None

    # Setup Intrinsic Reward Function if configured
    if config.intrinsic_reward is not None:
        intrinsic_reward_class: type[IntrinsicReward] = getattr(
            intrinsic_rewards, f"{config.intrinsic_reward}Reward"
        )

        intrinsic_reward = intrinsic_reward_class(
            device=device,
            prior_model_file_path=prior_model_file_path,
            learning_rate=config.learning_rate,
        )

        logger.info(f"Using intrinsic reward: {intrinsic_reward.__class__.__name__}")

    return diversity_filter(
        bucket_size=config.bucket_size,
        minscore=config.minscore,
        minsimilarity=config.minsimilarity,
        penalty_multiplier=config.penalty_multiplier,
        rdkit_smiles_flags=rdkit_smiles_flags,
        discard=config.discard,
        merge_threshold=config.merge_threshold,
        branching_factor=config.branching_factor,
        recluster_tolerance=config.recluster_tolerance,
        recluster_interval=config.recluster_interval,
        learning_rate=config.learning_rate,
        penalty_function=penalty,
        intrinsic_reward=intrinsic_reward,
    )
