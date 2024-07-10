"""Nomralize SMILES

Implemented as a decorator.
"""

__all__ = ["normalize_smiles"]

from typing import List, Callable
import logging

from . import normalizers

logger = logging.getLogger("reinvent")


def normalize_smiles(func: Callable):
    def wrapper(self, smilies: List[str]):
        normalizer = getattr(normalizers, self.smiles_type)

        cleaned_smilies = normalizer.normalize(smilies)

        return func(self, cleaned_smilies)

    return wrapper
