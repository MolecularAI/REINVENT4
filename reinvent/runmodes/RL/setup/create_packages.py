from __future__ import annotations

from typing import List
import logging

from . import terminators
from .terminators import terminator_callable
from reinvent.runmodes import RL
from reinvent.runmodes.RL.data_classes import WorkPackage
from reinvent.runmodes.RL.setup.diversity_filter import setup_diversity_filter
from reinvent.runmodes.RL.validation import SectionStage
from reinvent.scoring import Scorer

logger = logging.getLogger(__name__)


def create_packages(
    reward_strategy: RL.RLReward, stages: List[SectionStage], rdkit_smiles_flags: dict
) -> List[WorkPackage]:
    """Create work packages

    Collect the stage parameters and build a work package for each stage.  The
    parameters are from a dict, so the keys (parameters) are hard-coded here.
    Each stage can define its own scoring function.

    :param reward_strategy: the reward strategy
    :param stages: the parameters for each work package
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: a list of work packages
    """
    packages = []

    for stage in stages:
        chkpt_filename = stage.chkpt_file

        scoring_function = Scorer(stage.scoring)

        max_score = stage.max_score
        min_steps = stage.min_steps
        max_steps = stage.max_steps

        terminator_param = stage.termination
        terminator_name = terminator_param.lower().title()

        try:
            terminator: terminator_callable = getattr(terminators, f"{terminator_name}Terminator")
        except KeyError:
            msg = f"Unknown termination criterion: {terminator_name}"
            logger.critical(msg)
            raise RuntimeError(msg)

        diversity_filter = None

        if stage.diversity_filter:
            diversity_filter = setup_diversity_filter(stage.diversity_filter, rdkit_smiles_flags)

        packages.append(
            WorkPackage(
                scoring_function,
                reward_strategy,
                max_steps,
                terminator(max_score, min_steps),
                diversity_filter,
                chkpt_filename,
            )
        )

    return packages
