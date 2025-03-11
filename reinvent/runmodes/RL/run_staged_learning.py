"""Multi-stage learning with RL"""

from __future__ import annotations
import os
import logging
from typing import List, TYPE_CHECKING

import torch

from reinvent.utils import setup_logger, CsvFormatter, config_parse
from reinvent.runmodes import Handler, RL, create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL import terminators, memories
from reinvent.runmodes.RL.data_classes import WorkPackage, ModelState
from reinvent.runmodes.utils import disable_gradients
from reinvent.scoring import Scorer
from .validation import RLConfig

if TYPE_CHECKING:
    from reinvent.runmodes.RL import terminator_callable
    from reinvent.models import ModelAdapter
    from .validation import (
        SectionDiversityFilter,
        SectionLearningStrategy,
        SectionInception,
        SectionStage,
    )

logger = logging.getLogger(__name__)

TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]


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


def setup_inception(config: SectionInception, prior: ModelAdapter):
    """Setup inception memory

    :param config: the config specific to the inception memory
    :param prior: the prior network
    :return: the set up inception memory or None
    """

    smilies = []
    deduplicate = config.deduplicate
    smilies_filename = config.smiles_file

    if smilies_filename and os.path.exists(smilies_filename):
        smilies = config_parse.read_smiles_csv_file(smilies_filename, columns=0)

        if not smilies:
            msg = f"Inception SMILES could not be read from {smilies_filename}"
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.info(f"Inception SMILES read from {smilies_filename}")

    if not smilies:
        logger.info(f"No SMILES for inception. Populating from first sampled batch.")

    if deduplicate:
        logger.info("Global SMILES deduplication for inception memory")

    inception = memories.Inception(
        memory_size=config.memory_size,
        sample_size=config.sample_size,
        smilies=smilies,
        scoring_function=None,
        prior=prior,
        deduplicate=deduplicate,
    )

    logger.info(f"Using inception memory")

    return inception


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


def run_staged_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str,
    responder_config: dict,
    write_config: str = None,
    *args,
    **kwargs,
):
    """Run Reinforcement Learning/Curriculum Learning

    RL can be run in multiple stages (CL) with different parameters, typically
    the scoring function.  A context manager ensures that a checkpoint file is
    written out should the program be terminated.

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: TensorBoard log directory
    :param responder_config: responder configuration
    :param write_config: callable to write config
    """

    config = RLConfig(**input_config)
    stages = config.stage
    num_stages = len(stages)
    logger.info(
        f"Starting {num_stages} {'stages' if num_stages> 1 else 'stage'} of Reinforcement Learning"
    )

    parameters = config.parameters
    
    # NOTE: The model files are a dictionary with model attributes from
    #       Reinvent and a set of tensors, each with an attribute for the
    #       device (CPU or GPU) and if gradients are required

    prior_model_filename = parameters.prior_file
    agent_model_filename = parameters.agent_file

    # NOTE: Inference mode means here that torch runs eval() on the network:
    #       switch off some specific layers (dropout, batch normal) but that
    #       does not affect autograd
    #       The gradients are switched off for the prior but the optimizer will
    #       not touch those anyway because we pass only the agent network to the
    #       optimizer, see above.
    adapter, _, model_type = create_adapter(prior_model_filename, "inference", device)
    prior = adapter
    disable_gradients(prior)

    rdkit_smiles_flags = dict(allowTautomers=True)

    if model_type in TRANSFORMERS:  # Transformer-based models
        agent_mode = "inference"
        rdkit_smiles_flags.update(sanitize=True, isomericSmiles=True)
        rdkit_smiles_flags2 = dict(isomericSmiles=True)
    else:
        agent_mode = "training"
        rdkit_smiles_flags2 = dict()

    adapter, agent_save_dict, agent_model_type = create_adapter(
        agent_model_filename, agent_mode, device
    )
    agent = adapter

    if model_type != agent_model_type:
        msg = f"Inconsistent model types: prior is {model_type} agent is {agent_model_type}"
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info(f"Using generator {model_type}")
    logger.info(f"Prior read from {prior_model_filename}")
    logger.info(f"Agent read from {agent_model_filename}")

    smilies = None

    if parameters.smiles_file:
        smilies = config_parse.read_smiles_csv_file(parameters.smiles_file, columns=0)
        logger.info(f"Input molecules/fragments read from file {parameters.smiles_file}")

    sampler, _ = setup_sampler(model_type, parameters.dict(), agent)
    reward_strategy = setup_reward_strategy(config.learning_strategy, agent)

    global_df_only = False

    if parameters.use_checkpoint and "staged_learning" in agent_save_dict:
        logger.info(f"Using diversity filter from {agent_model_filename}")
        diversity_filter = agent_save_dict["staged_learning"]["diversity_filter"]
    elif config.diversity_filter:
        diversity_filter = setup_diversity_filter(config.diversity_filter, rdkit_smiles_flags2)
        global_df_only = True

    if parameters.purge_memories:
        logger.info("Purging diversity filter memories after each stage")
    else:
        logger.info("Diversity filter memories are retained between stages")

    inception = None

    # Inception only set up for the very first step
    if config.inception and model_type == "Reinvent":
        inception = setup_inception(config.inception, prior)

    if not inception and model_type == "Reinvent":
        logger.warning("Inception disabled but may speed up convergence")

    packages = create_packages(reward_strategy, stages, rdkit_smiles_flags2)

    summary_csv_prefix = parameters.summary_csv_prefix

    # FIXME: is there a sensible default, this is only needed by Mol2Mol
    distance_threshold = parameters.distance_threshold

    model_learning = getattr(RL, f"{model_type}Learning")
    
    if callable(write_config):
        write_config(config.model_dump())

    with Handler() as handler:
        for run, package in enumerate(packages):
            stage_no = run + 1
            csv_filename = f"{summary_csv_prefix}_{stage_no}.csv"

            setup_logger(
                name="csv",
                filename=csv_filename,
                formatter=CsvFormatter(),
                propagate=False,
                level="INFO",
            )

            logdir = f"{tb_logdir}_{run}" if tb_logdir else None

            logger.info(f"Writing tabular data for stage to {csv_filename}")
            logger.info(f"Starting stage {stage_no} <<<")

            if global_df_only:  # global DF always overwrites stage DFs
                state = ModelState(agent, diversity_filter)
                logger.debug(f"Using global DF")
            else:
                state = ModelState(agent, package.diversity_filter)
                logger.debug(f"Using stage DF")
            
            optimize = model_learning(
                max_steps=package.max_steps,
                stage_no=stage_no,
                prior=prior,
                state=state,
                scoring_function=package.scoring_function,
                reward_strategy=package.learning_strategy,
                sampling_model=sampler,
                smilies=smilies,
                distance_threshold=distance_threshold,
                rdkit_smiles_flags=rdkit_smiles_flags,
                inception=inception,
                responder_config=responder_config,
                tb_logdir=logdir,
                tb_isim=parameters.tb_isim,
            )

            if device.type == "cuda" and torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                free_memory //= 1024**2
                used_memory = total_memory // 1024**2 - free_memory
                logger.info(
                    f"Current GPU memory usage: {used_memory} MiB used, {free_memory} MiB free"
                )

            handler.out_filename = package.out_state_filename
            handler.register_callback(optimize.get_state_dict)

            if inception:
                inception.update_scoring_function(package.scoring_function)

            terminate = optimize(package.terminator)
            state = optimize.state

            if state.diversity_filter and parameters.purge_memories:
                logger.info(f"Purging diversity filter memories in stage {stage_no}")
                state.diversity_filter.purge_memories()

            handler.save()

            if terminate:
                logger.warning(
                    f"Maximum number of steps of {package.max_steps} reached in stage "
                    f"{stage_no}. Terminating all stages."
                )
                break

            logger.info(f"Finished stage {stage_no} >>>")
