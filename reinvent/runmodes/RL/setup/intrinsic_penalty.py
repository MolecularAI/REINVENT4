from __future__ import annotations

import logging

from reinvent.runmodes.RL import intrinsic_penalty
from reinvent.runmodes.RL.validation import SectionIntrinsicPenalty

logger = logging.getLogger(__name__)


def setup_intrinsic_penalty(
    config: SectionIntrinsicPenalty,
    device: torch.device,
    prior_model_file_path: str,
    rdkit_smiles_flags: dict,
):
    """Setup of the intrinsic penalty

    Basic setup of the intrinsic penalty memory. The parameters are from a
    dict, so the keys (parameters) are hard-coded here.

    :param config: config parameter specific to the filter
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :param device: device to run any intrinsic reward model on
    :param prior_model_file_path: path to prior model file
    :return: the set up diversity filter
    """

    if config is None or not hasattr(config, "type"):
        return None

    diversity_filter = getattr(intrinsic_penalty, config.type)

    logger.info(f"Using intrinsic penalty {config.type}")

    return diversity_filter(
        penalty_function=config.penalty_function,
        bucket_size=config.bucket_size,
        minscore=config.minscore,
        learning_rate=config.learning_rate,
        device=device,
        prior_model_file_path=prior_model_file_path,
        rdkit_smiles_flags=rdkit_smiles_flags,
    )
