"""Sampler setup needed for sampling and staged learning"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import warnings

from reinvent.runmodes import samplers
from reinvent.chemistry import TransformationTokens

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter
    from reinvent.runmodes.dtos import ChemistryHelpers

logger = logging.getLogger(__name__)
warnings.filterwarnings("once", category=FutureWarning)

TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer"]

def setup_sampler(model_type: str, config: dict, agent: ModelAdapter, chemistry: ChemistryHelpers):
    """Setup the sampling module.

    The sampler module must be for the same model as the optimization strategy.

    FIXME: make sure sampler and optimization strategy always match.

    :param model_type: name of the model type
    :param config: the config specific to the sampler
    :param agent: the agent model network
    :param chemistry: a namespace holding the various chemistry helpers
    :return: the set up sampler
    """

    # number of smiles to be generated for each input;
    # different from batch size used in dataloader which affect cuda memory
    batch_size = config.get("batch_size", 100)
    randomize_smiles = config.get("randomize_smiles", True)
    temperature = config.get("temperature", 1.0)

    # Transformer-based models were trained on canonical SMILES
    if model_type in TRANSFORMERS and randomize_smiles:
        randomize_smiles = False
        logger.warning(f"randomize_smiles is set to be True by user. But the model was trained using canonical SMILES"
                       f"where randomize_smiles might undermine the performance (this needs more investigation), "
                       f"but randomize_smiles is reset to be False for now.")

    unique_sequences = config.get("unique_sequences", False)

    if unique_sequences:
        warnings.warn(
            "Sequence deduplication is deprecated and will be removed in the future.",
            FutureWarning,
            stacklevel=2,
        )

    if model_type in TRANSFORMERS:
        sample_strategy = config.get("sample_strategy", "multinomial")
    else:
        sample_strategy = None

    tokens = TransformationTokens()  # LinkInvent only
    isomeric = False

    if model_type in TRANSFORMERS:  # for Transformer-based models
        isomeric = True

        agent.model.set_temperature(temperature)
        if sample_strategy == "beamsearch":
            agent.model.set_beam_size(batch_size)

    sampling_model = getattr(samplers, f"{model_type}Sampler")
    sampler = sampling_model(
        agent,
        batch_size=batch_size,
        sample_strategy=sample_strategy,  # needed for Transformer-based models
        isomeric=isomeric,  # needed for Transformer-based models
        randomize_smiles=randomize_smiles,
        unique_sequences=unique_sequences,
        chemistry=chemistry,
        tokens=tokens,
    )

    return sampler, batch_size
