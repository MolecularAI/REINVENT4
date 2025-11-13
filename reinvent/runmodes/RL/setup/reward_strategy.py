from __future__ import annotations

import logging

import torch

from reinvent.models import ModelAdapter
from reinvent.runmodes import RL
from reinvent.runmodes.RL.validation import SectionLearningStrategy

logger = logging.getLogger(__name__)


def setup_reward_strategy(config: SectionLearningStrategy, agent: ModelAdapter):
    """Setup the Reinforcement Learning reward strategy

    Basic parameter setup for RL learning including the reward function. The
    parameters are from a dict, so the keys (parameters) are hard-coded here.

    DAP has been found to be the best choice, see https://doi.org/10.1021/acs.jcim.1c00469.
    SDAP seems to have a smaller learning rate while the other two (MAULI, MASCOF)
    do not seem to bes useful at all.

    :param config: the configuration
    :param agent: the agent model network
    :return: the set up RL strategy
    """

    learning_rate = config.rate
    sigma = config.sigma  # determines how dominant the score is

    reward_strategy_str = config.type

    try:
        reward_strategy = getattr(RL, f"{reward_strategy_str}_strategy")
    except AttributeError:
        msg = f"Unknown reward strategy {reward_strategy_str}"
        logger.critical(msg)
        raise RuntimeError(msg)

    torch_optim = torch.optim.Adam(agent.get_network_parameters(), lr=learning_rate)
    learning_strategy = RL.RLReward(torch_optim, sigma, reward_strategy)

    logger.info(f"Using reward strategy {reward_strategy_str}")

    return learning_strategy
