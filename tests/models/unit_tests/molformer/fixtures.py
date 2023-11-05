import torch.nn as nn

from reinvent.models.mol2mol.models.encode_decode.model import EncoderDecoder
from reinvent.models.mol2mol.models.vocabulary import Vocabulary
from reinvent.models.mol2mol.model import Mol2MolModel


def _init_params(parameters):
    """
    Fixed weights
    """
    for p in parameters:
        if p.dim() > 1:
            nn.init.constant_(p, 0.5)


def mocked_molformer_model():
    vocabulary = Vocabulary({"*": 0, "^": 1, "$": 2, "c": 3, "O": 4, "C": 5, "N": 6, "1": 7})
    encoder_decoder = EncoderDecoder(len(vocabulary))
    model = Mol2MolModel(vocabulary, encoder_decoder)
    _init_params(model.network.parameters())
    return model


def mocked_vocabulary():
    vocabulary = Vocabulary({"*": 0, "^": 1, "$": 2, "c": 3, "O": 4, "C": 5, "N": 6, "1": 7})
    return vocabulary
