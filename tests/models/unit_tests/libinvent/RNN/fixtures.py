import torch.nn as nn

from reinvent.models.model_mode_enum import ModelModeEnum
from reinvent.models.model_parameter_enum import ModelParametersEnum
from reinvent.models.libinvent.models.decorator import Decorator
from reinvent.models.libinvent.models.model import DecoratorModel
from reinvent.models.libinvent.models.vocabulary import DecoratorVocabulary
from reinvent.models import meta_data
from tests.test_data import SCAFFOLD_SUZUKI


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_decorator_model():
    smiles_list = [SCAFFOLD_SUZUKI]
    decorator_vocabulary = DecoratorVocabulary.from_lists(smiles_list, smiles_list)
    scaffold_vocabulary_size = decorator_vocabulary.len_scaffold()
    decoration_vocabulary_size = decorator_vocabulary.len_decoration()

    parameter_enums = ModelParametersEnum
    encoder_params = {
        parameter_enums.NUMBER_OF_LAYERS: 2,
        parameter_enums.NUMBER_OF_DIMENSIONS: 128,
        parameter_enums.VOCABULARY_SIZE: scaffold_vocabulary_size,
        parameter_enums.DROPOUT: 0,
    }
    decoder_params = {
        parameter_enums.NUMBER_OF_LAYERS: 2,
        parameter_enums.NUMBER_OF_DIMENSIONS: 128,
        parameter_enums.VOCABULARY_SIZE: decoration_vocabulary_size,
        parameter_enums.DROPOUT: 0,
    }
    decorator = Decorator(encoder_params, decoder_params)

    model_regime = ModelModeEnum()

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = DecoratorModel(decorator_vocabulary, decorator, metadata, mode=model_regime.INFERENCE)
    _init_params(model.network.parameters())

    return model
