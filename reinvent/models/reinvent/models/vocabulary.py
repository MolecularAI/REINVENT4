"""Vocabulary and token handling
"""

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

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

    def add(self, token):
        """Adds a token."""

        if not isinstance(token, str):
            raise TypeError("Token is not a string")

        if token in self:
            return self[token]

        self._add(token, self._current_id)
        self._current_id += 1

        return self._current_id - 1

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def update(self, tokens):
        """Adds many tokens."""

        return [self.add(token) for token in tokens]

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)

        for i, token in enumerate(tokens):
            if token not in self._tokens:
                msg = f"unknown token {token}: {tokens}"
                logger.critical(msg)
                raise RuntimeError(msg)

            vocab_index[i] = self._tokens[token]

        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

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


REGEXPS = {
    "brackets": re.compile(r"(\[[^\]]*\])"),
    "2_ring_nums": re.compile(r"(%\d{2})"),
    "brcl": re.compile(r"(Br|Cl)"),
}
REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
# FIXME: those two are shared with other generators
START_TOKEN = "^"
STOP_TOKEN = "$"


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        tokens = split_by(data, REGEXP_ORDER)

        if with_begin_and_end:
            tokens = [START_TOKEN] + tokens + [STOP_TOKEN]

        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""

        smi = ""

        for token in tokens:
            if token == STOP_TOKEN:
                break

            if token != START_TOKEN:
                smi += token

        return smi


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()

    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    # stop token is 0 (also counts as padding)
    vocabulary.update([STOP_TOKEN, START_TOKEN] + sorted(tokens))

    return vocabulary


def split_by(data, regexps):
    if not regexps:
        return list(data)

    regexp = REGEXPS[regexps[0]]
    splitted = regexp.split(data)
    tokens = []

    for i, split in enumerate(splitted):
        if i % 2 == 0:
            tokens += split_by(split, regexps[1:])
        else:
            tokens.append(split)

    return tokens
