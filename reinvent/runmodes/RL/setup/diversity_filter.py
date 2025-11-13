from __future__ import annotations

import logging

from reinvent.runmodes.RL import memories
from reinvent.runmodes.RL.validation import SectionDiversityFilter

logger = logging.getLogger(__name__)


def setup_diversity_filter(config: SectionDiversityFilter, rdkit_smiles_flags: dict):
    """Setup of the diversity filter

    Basic setup of the diversity filter memory.  The parameters are from a
    dict, so the keys (parameters) are hard-coded here.

    :param config: config parameter specific to the filter
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: the set up diversity filter
    """

    if config is None or not hasattr(config, "type"):
        return None

    diversity_filter = getattr(memories, config.type)

    logger.info(f"Using diversity filter {config.type}")

    return diversity_filter(
        bucket_size=config.bucket_size,
        minscore=config.minscore,
        minsimilarity=config.minsimilarity,
        penalty_multiplier=config.penalty_multiplier,
        rdkit_smiles_flags=rdkit_smiles_flags,
    )
