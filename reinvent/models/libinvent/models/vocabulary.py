# coding=utf-8

"""
Vocabulary helper class
"""

import re

import numpy as np


class Vocabulary:
    """Stores the tokens and their conversion to one-hot vectors."""

    def __init__(self, tokens=None, starting_id=0):
        """
        Instantiates a Vocabulary instance.
        :param tokens: A list of tokens (str).
        :param starting_id: The value for the starting id.
        :return:
        """
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        """
        Retrieves the if the token is given or a token if the id is given.
        :param token_or_id: A token or an id.
        :return: An id if a token was given or a token if an id was given.
        """
        return self._tokens[token_or_id]

    def add(self, token):
        """
        Adds a token to the vocabulary.
        :param token: Token to add.
        :return: The id assigned to the token. If the token was already there,
                 the id of that token is returned instead.
        """
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """
        Adds many tokens at once.
        :param tokens: A list of tokens.
        :return: The ids of the tokens added.
        """
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        """
        Deletes a (token, id) tuple, given a token or an id.
        :param token_or_id: A token or an id.
        :return:
        """
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        """
        Checks whether a token is contained in the vocabulary.
        :param a token or an id to check
        :return : True if it is contained, otherwise False.
        """
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        """
        Compares two vocabularies.
        :param other_vocabulary: Other vocabulary to be checked.
        :return: True if they are the same.
        """
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        """
        Calculates the length (number of tokens) of the vocabulary.
        :return : The number of tokens.
        """
        return len(self._tokens) // 2

    def encode(self, tokens):
        """
        Encodes a list of tokens, encoding them in 1-hot encoded vectors.
        :param tokens: Tokens to encode.
        :return : An numpy array with the tokens encoded.
        """

        ohe_vect = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            if token in self._tokens:
                ohe_vect[i] = self._tokens[token]
            else:
                raise KeyError(f"{token} is not supported! Supported tokens are {self.tokens()}.")
        return ohe_vect

    def decode(self, ohe_vect):
        """
        Decodes a one-hot encoded vector matrix to a list of tokens.
        :param : A numpy array with some encoded tokens.
        :return : An unencoded version of the input array.
        """
        tokens = []
        for ohv in ohe_vect:
            tokens.append(self[ohv])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """
        Returns the tokens from the vocabulary.
        :return: A list of tokens.
        """
        return [t for t in self._tokens if isinstance(t, str)]


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, smiles, with_begin_and_end=True):
        """
        Tokenizes a SMILES string.
        :param smiles: A SMILES string.
        :param with_begin_and_end: Appends a begin token and prepends an end token.
        :return : A list with the tokenized version.
        """

        def split_by(smiles, regexps):
            if not regexps:
                return list(smiles)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(smiles, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """
        Untokenizes a SMILES string.
        :param tokens: List of tokens.
        :return : A SMILES string.
        """
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer):
    """
    Creates a vocabulary for the SMILES syntax.
    :param smiles_list: A list with SMILES.
    :param tokenizer: Tokenizer to use.
    :return: A vocabulary instance with all the tokens in the smiles_list.
    """
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["<pad>", "$", "^"] + sorted(tokens))
    return vocabulary


class DecoratorVocabulary:
    """
    Encapsulation of the two vocabularies needed for the decorator.
    """

    def __init__(
        self, scaffold_vocabulary, scaffold_tokenizer, decoration_vocabulary, decoration_tokenizer
    ):
        self.scaffold_vocabulary = scaffold_vocabulary
        self.scaffold_tokenizer = scaffold_tokenizer
        self.decoration_vocabulary = decoration_vocabulary
        self.decoration_tokenizer = decoration_tokenizer

    def len_scaffold(self):
        """
        Returns the length of the scaffold vocabulary.
        """
        return len(self.scaffold_vocabulary)

    def len_decoration(self):
        """
        Returns the length of the decoration vocabulary.
        """
        return len(self.decoration_vocabulary)

    def len(self):
        """
        Returns the lenght of both vocabularies in a tuple.
        :return: A tuple with (len(scaff_voc), len(dec_voc)).
        """
        return (len(self.scaffold_vocabulary), len(self.decoration_vocabulary))

    def encode_scaffold(self, smiles):
        """
        Encodes a scaffold SMILES.
        :param smiles: Scaffold SMILES to encode.
        :return : An one-hot-encoded vector with the scaffold information.
        """
        return self.scaffold_vocabulary.encode(self.scaffold_tokenizer.tokenize(smiles))

    def decode_scaffold(self, encoded_scaffold):
        """
        Decodes the scaffold.
        :param encoded_scaffold: A one-hot encoded version of the scaffold.
        :return : A SMILES of the scaffold.
        """
        return self.scaffold_tokenizer.untokenize(self.scaffold_vocabulary.decode(encoded_scaffold))

    def encode_decoration(self, smiles):
        """
        Encodes a decoration SMILES.
        :param smiles: Decoration SMILES to encode.
        :return : An one-hot-encoded vector with the fragment information.
        """
        return self.decoration_vocabulary.encode(self.decoration_tokenizer.tokenize(smiles))

    def decode_decoration(self, encoded_decoration):
        """
        Decodes the decorations for a scaffold.
        :param encoded_decorations: A one-hot encoded version of the decoration.
        :return : A list with SMILES of all the fragments.
        """
        return self.decoration_tokenizer.untokenize(
            self.decoration_vocabulary.decode(encoded_decoration)
        )

    @classmethod
    def from_lists(cls, scaffold_list, decoration_list):
        """
        Creates the vocabularies from lists.
        :param scaffold_list: A list with scaffolds.
        :param decoration_list: A list with decorations.
        :return : A DecoratorVocabulary instance
        """
        scaffold_tokenizer = SMILESTokenizer()
        scaffold_vocabulary = create_vocabulary(scaffold_list, scaffold_tokenizer)

        decoration_tokenizer = SMILESTokenizer()
        decoration_vocabulary = create_vocabulary(decoration_list, decoration_tokenizer)

        return cls(
            scaffold_vocabulary, scaffold_tokenizer, decoration_vocabulary, decoration_tokenizer
        )
