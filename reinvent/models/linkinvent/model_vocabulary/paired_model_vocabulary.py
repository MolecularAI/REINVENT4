"""Vocabulary for LinkInvent.

Input is a synonym for warheads (two separated by '|')
Output is a synonym for the linker, also named target.
"""

from typing import List

from reinvent.models.linkinvent.model_vocabulary.model_vocabulary import ModelVocabulary
from reinvent.models.linkinvent.model_vocabulary.vocabulary import SMILESTokenizer, Vocabulary


class PairedModelVocabulary:
    def __init__(
        self,
        input_vocabulary: Vocabulary,
        input_tokenizer: SMILESTokenizer,
        output_vocabulary: Vocabulary,
        output_tokenizer: SMILESTokenizer,
    ):
        self.input = ModelVocabulary(input_vocabulary, input_tokenizer)
        self.target = ModelVocabulary(output_vocabulary, output_tokenizer)

    def len(self):
        """
        Returns the length of both input and output vocabulary in a tuple

        :return: len(input_vocabulary), len(output_vocabulary)

        """
        return len(self.input), len(self.target)

    @classmethod
    def from_lists(cls, input_smiles_list: List[str], target_smiles_list: List[str]):
        input_vocabulary = ModelVocabulary.from_list(input_smiles_list)
        target_vocabulary = ModelVocabulary.from_list(target_smiles_list)

        return cls(
            input_vocabulary.vocabulary,
            input_vocabulary.tokenizer,
            target_vocabulary.vocabulary,
            input_vocabulary.tokenizer,
        )
