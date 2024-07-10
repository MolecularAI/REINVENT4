"""Vocabulary helper class"""

import re

import numpy as np


class Vocabulary:
    """Stores the tokens and their conversion to one-hot vectors."""

    def __init__(
        self, tokens=None, starting_id=0, pad_token=0, bos_token=1, eos_token=2, unk_token=None
    ):
        self._tokens = {}
        self._current_id = starting_id

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
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
        """Adds many tokens."""
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
        :param token_or_id: token or an id to check
        :return : True if it is contained, otherwise False.
        """
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        """
        Compares two vocabularies.
        :param other_vocabulary: Other vocabulary to be checked.
        :return: True if they are the same.
        """
        return self._tokens == other_vocabulary._tokens

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
        ohe_keep_mask = np.ones_like(tokens, dtype=bool)
        for i, token in enumerate(tokens):
            if token not in self._tokens:
                if hasattr(self, "unk_token") and (self.unk_token is not None):
                    unk_symbol = self[self.unk_token]
                    ohe_vect[i] = self._tokens[unk_symbol]
                else:
                    ohe_keep_mask[i] = False
            else:
                ohe_vect[i] = self._tokens[token]
        return ohe_vect[ohe_keep_mask]

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
            raise ValueError(f"Index {idx} already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]

    def word2idx(self):
        return {k: self._tokens[k] for k in self._tokens if isinstance(k, str)}

    def get_dictionary(self):
        return {
            "tokens": self.word2idx(),
            "pad_token": getattr(self, "pad_token", 0),
            "bos_token": getattr(self, "bos_token", 1),
            "eos_token": getattr(self, "eos_token", 2),
            "unk_token": getattr(self, "unk_token", None),
        }

    @classmethod
    def load_from_dictionary(cls, dictionary: dict):
        vocabulary = cls()
        for k, i in dictionary["tokens"].items():
            vocabulary._add(str(k), int(i))
        vocabulary.pad_token = dictionary["pad_token"]
        vocabulary.bos_token = dictionary["bos_token"]
        vocabulary.eos_token = dictionary["eos_token"]
        vocabulary.unk_token = dictionary["unk_token"]
        return vocabulary


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^]]*])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """
        Tokenizes a SMILES string.
        :param with_begin_and_end: Appends a begin token and prepends an end token.
        :return : A list with the tokenized version.
        """

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
        """
        Untokenizes a SMILES string.
        :param tokens: List of tokens.
        :return : A SMILES string.
        """
        smi = ""
        for i, token in enumerate(tokens):
            if token == "$":
                break
            if token != "^" or (token == "^" and i != 0):
                smi += token
        return smi


# for linkinvent and libinvent
def build_vocabulary(
    smiles_list, tokenizer=SMILESTokenizer(), add_unused=False, num_unused_tokens=50
) -> Vocabulary:
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
    vocabulary.update(["<PAD>", "^", "$", "<UNK>"] + sorted(tokens))  # pad=0, start=1, end=2

    if add_unused:
        unnsed_tokens = [f"Unused{i+1}" for i in range(num_unused_tokens)]
        vocabulary.update(unnsed_tokens)

    vocabulary.pad_token = 0  # 0 is padding
    vocabulary.bos_token = 1  # 1 is start symbol
    vocabulary.eos_token = 2  # 2 is end symbol
    vocabulary.unk_token = 3  # 3 is an unknown symbol
    return vocabulary


# Todo clean, check usage
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
