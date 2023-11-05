from typing import List

from reinvent.models.linkinvent.model_vocabulary.vocabulary import (
    Vocabulary,
    SMILESTokenizer,
    create_vocabulary,
)


class ModelVocabulary:
    def __init__(self, vocabulary: Vocabulary, tokenizer: SMILESTokenizer):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """ " Returns the length of the vocabulary"""
        return len(self.vocabulary)

    def encode(self, smiles_str: str):
        """
        Encode a smiles str

        :param smiles_str:
        :return: An one-hot-encoded vector
        """
        return self.vocabulary.encode(self.tokenizer.tokenize(smiles_str))

    def decode(self, encoded_vector) -> str:
        """
        Decodes the encoded vector.

        :param encoded_vector: A one-hot encoded version of the target.
        :return : A SMILES of the target.
        """
        return self.tokenizer.untokenize(self.vocabulary.decode(encoded_vector))

    @classmethod
    def from_list(cls, smiles_list: List[str]):
        """
        Creates the ModelVocabulary form a list of smiles

        :param smiles_list: A list of smiles
        :return : A ModelVocabulary instance

        """
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer)
        return cls(vocabulary, tokenizer)
