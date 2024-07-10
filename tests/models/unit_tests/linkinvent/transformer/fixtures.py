import pytest
from torch import nn

from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import build_vocabulary
from reinvent.models.transformer.linkinvent.linkinvent import LinkinventModel
from reinvent.models import meta_data
from tests.conftest import device
from tests.test_data import (
    WARHEAD_PAIR,
    WARHEAD_TRIPLE,
    WARHEAD_QUADRUPLE,
    SCAFFOLD_TO_DECORATE,
    LINKER_TRIPLE,
)


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_linkinvent_model():
    vocabulary = mocked_vocabulary()
    encoder_decoder = EncoderDecoder(len(vocabulary))

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = LinkinventModel(vocabulary, encoder_decoder, metadata)
    _init_params(model.network.parameters())
    return model


def mocked_vocabulary():
    smiles_list = [
        WARHEAD_PAIR,
        WARHEAD_TRIPLE,
        WARHEAD_QUADRUPLE,
        SCAFFOLD_TO_DECORATE,
        LINKER_TRIPLE,
    ]
    vocabulary = build_vocabulary(smiles_list)

    return vocabulary
