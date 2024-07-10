"""SMILES dataset for PyTorch DataLoader"""

import torch
import torch.utils.data as tud

from reinvent.chemistry import conversions
from reinvent.models.reinvent.utils import collate_fn


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __new__(cls, *args, **kwargs):
        if "randomize" in kwargs and kwargs["randomize"]:
            cls.__getitem__ = cls._getitem_with_randomization
        else:
            cls.__getitem__ = cls._getitem

        return super().__new__(cls)

    def __init__(self, smiles_list, vocabulary, tokenizer, randomize=False):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._smiles_list = list(smiles_list)

    def _getitem(self, i):
        smiles = self._smiles_list[i]
        tokens = self._tokenizer.tokenize(smiles)
        encoded = self._vocabulary.encode(tokens)

        return torch.tensor(encoded, dtype=torch.long)

    def _getitem_with_randomization(self, i):
        smiles = conversions.randomize_smiles(self._smiles_list[i])
        tokens = self._tokenizer.tokenize(smiles)
        encoded = self._vocabulary.encode(tokens)

        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self._smiles_list)


Dataset.collate_fn = collate_fn
