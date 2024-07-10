import torch.nn as nn

from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import Vocabulary
from reinvent.models import meta_data


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_mol2mol_model():
    vocabulary = mocked_vocabulary()
    encoder_decoder = EncoderDecoder(len(vocabulary))

    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )

    model = Mol2MolModel(vocabulary, encoder_decoder, metadata)
    _init_params(model.network.parameters())
    return model


def mocked_vocabulary():
    vocabulary = Vocabulary({"*": 0, "^": 1, "$": 2, "c": 3, "O": 4, "C": 5, "N": 6, "1": 7})
    return vocabulary
