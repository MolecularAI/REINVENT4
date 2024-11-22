import pytest
from torch import nn

from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import build_vocabulary
from reinvent.models.transformer.pepinvent.pepinvent import PepinventModel
from reinvent.models import meta_data
from tests.test_data import PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_INPUT3


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_pepinvent_model():
    vocabulary = mocked_vocabulary()
    encoder_decoder = EncoderDecoder(len(vocabulary))

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = PepinventModel(vocabulary, encoder_decoder, metadata)
    _init_params(model.network.parameters())
    return model


def mocked_vocabulary():
    smiles_list = [
        PEPINVENT_INPUT1, PEPINVENT_INPUT2, PEPINVENT_INPUT3
    ]
    vocabulary = build_vocabulary(smiles_list)

    return vocabulary
