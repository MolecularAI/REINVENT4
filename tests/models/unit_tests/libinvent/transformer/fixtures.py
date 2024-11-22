import pytest
from torch import nn

from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import build_vocabulary
from reinvent.models.transformer.libinvent.libinvent import LibinventModel
from reinvent.models import meta_data
from tests.conftest import device
from tests.test_data import SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, \
    SCAFFOLD_QUADRUPLE_POINT, DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_libinvent_model():
    vocabulary = mocked_vocabulary()
    encoder_decoder = EncoderDecoder(len(vocabulary))

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = LibinventModel(vocabulary, encoder_decoder, metadata)
    _init_params(model.network.parameters())
    return model


def mocked_vocabulary():
    smiles_list = [
        SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, SCAFFOLD_QUADRUPLE_POINT,
        DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS
    ]
    vocabulary = build_vocabulary(smiles_list)

    return vocabulary
