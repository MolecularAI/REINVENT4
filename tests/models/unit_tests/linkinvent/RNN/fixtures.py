from reinvent.models.linkinvent.link_invent_model import LinkInventModel
from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)
from reinvent.models.linkinvent.networks import EncoderDecoder
from reinvent.models.model_mode_enum import ModelModeEnum
from reinvent.models.model_parameter_enum import ModelParametersEnum
from reinvent.models import meta_data
from tests.test_data import WARHEAD_PAIR, ETHANE, HEXANE, PROPANE, BUTANE


def mocked_linkinvent_model():
    input_list = [WARHEAD_PAIR]
    output_list = [ETHANE, HEXANE, PROPANE, BUTANE]

    vocabulary = PairedModelVocabulary.from_lists(input_list, output_list)
    input_vocabulary_size, output_vocabulary_size = vocabulary.len()

    parameter_enums = ModelParametersEnum
    encoder_params = {
        parameter_enums.NUMBER_OF_LAYERS: 2,
        parameter_enums.NUMBER_OF_DIMENSIONS: 128,
        parameter_enums.VOCABULARY_SIZE: input_vocabulary_size,
        parameter_enums.DROPOUT: 0,
    }
    decoder_params = {
        parameter_enums.NUMBER_OF_LAYERS: 2,
        parameter_enums.NUMBER_OF_DIMENSIONS: 128,
        parameter_enums.VOCABULARY_SIZE: output_vocabulary_size,
        parameter_enums.DROPOUT: 0,
    }
    network = EncoderDecoder(encoder_params, decoder_params)

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = LinkInventModel(vocabulary, network, metadata, mode=ModelModeEnum().INFERENCE)
    return model
