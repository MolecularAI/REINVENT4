# coding=utf-8

"""
Vocabulary helper class
"""

import re

import numpy as np


class Vocabulary:
    """Stores the tokens and their conversion to one-hot vectors."""

    def __init__(self, tokens=None, starting_id=0, padding_value=0):
        self._tokens = {}
        self._current_id = starting_id
        self._padding_value = padding_value

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            raise ValueError("Token already present in the vocabulary")
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens, encoding them in 1-hot encoded vectors."""
        ohe_vect = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            ohe_vect[i] = self._tokens[token]
        return ohe_vect

    def decode(self, ohe_vect):
        """Decodes a one-hot encoded vector matrix to a list of tokens."""
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
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]

    def word2idx(self):
        return {k: self._tokens[k] for k in self._tokens if isinstance(k, str)}


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for i, token in enumerate(tokens):
            if token == "$":
                break
            if token != "^" or (token == "^" and i != 0):
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer, property_condition=None):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["*", "^", "$"] + sorted(tokens))  # pad=0, start=1, end=2
    if property_condition is not None:
        vocabulary.update(property_condition)
    # for random smiles
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])

    return vocabulary
