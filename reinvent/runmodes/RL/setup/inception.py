from __future__ import annotations

import os
import logging

from reinvent.models import ModelAdapter
from reinvent.runmodes.RL import memories
from reinvent.runmodes.RL.validation import SectionInception
from reinvent.utils import get_tokens_from_vocabulary, config_parse

logger = logging.getLogger(__name__)


def setup_inception(config: SectionInception, prior: ModelAdapter):
    """Setup inception memory

    :param config: the config specific to the inception memory
    :param prior: the prior network
    :return: the set up inception memory or None
    """

    smilies = []
    smilies_filename = config.smiles_file

    # FIXME: Lib- and Linkinvent would need to construct the molecule from the fragments
    if prior.model_type == "Reinvent" and smilies_filename and os.path.exists(smilies_filename):
        allowed_tokens = get_tokens_from_vocabulary(prior.vocabulary)

        # FIXME: this won't work for Lib- and Linkinvent
        smilies = config_parse.read_smiles_csv_file(
            smilies_filename, 0, allowed_tokens, remove_duplicates=True
        )

        if not smilies:
            msg = f"Inception SMILES could not be read from {smilies_filename}"
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.info(f"Inception SMILES read from {smilies_filename}")

    if not smilies:
        logger.info(f"No SMILES for inception. Populating from first sampled batch.")

    inception = memories.Inception(
        memory_size=config.memory_size,
        sample_size=config.sample_size,
        seed_smilies=smilies,
        scoring_function=None,
        prior=prior,
    )

    logger.info(f"Using inception memory")

    return inception
