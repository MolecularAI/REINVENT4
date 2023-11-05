# coding=utf-8
import torch
import torch.nn.utils.rnn as tnnur
import torch.utils.data as tud


class Dataset(tud.Dataset):
    """Dataset that takes a list of SMILES only."""

    def __init__(self, smiles_list, vocabulary, tokenizer):
        """
        Instantiates a Dataset.
        :param smiles_list: A list with SMILES strings.
        :param vocabulary: A Vocabulary object.
        :param tokenizer: A Tokenizer object.
        :return:
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_list = []
        for smi in smiles_list:
            tokenized = self._tokenizer.tokenize(smi)
            enc = self._vocabulary.encode(tokenized)

            if enc is not None:
                self._encoded_list.append(enc)

    def __getitem__(self, i):
        return torch.tensor(self._encoded_list[i], dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @staticmethod
    def collate_fn(encoded_seqs):
        return pad_batch(encoded_seqs)


class DecoratorDataset(tud.Dataset):
    """Dataset that takes a list of (scaffold, decoration) pairs."""

    def __init__(self, scaffold_decoration_smi_list, vocabulary):
        self.vocabulary = vocabulary
        self._encoded_list = []

        for scaffold, dec in scaffold_decoration_smi_list:
            en_scaff = self.vocabulary.scaffold_vocabulary.encode(
                self.vocabulary.scaffold_tokenizer.tokenize(scaffold)
            )
            en_dec = self.vocabulary.decoration_vocabulary.encode(
                self.vocabulary.decoration_tokenizer.tokenize(dec)
            )
            if en_scaff is not None and en_dec is not None:
                self._encoded_list.append((en_scaff, en_dec))

    def __getitem__(self, i):
        scaff, dec = self._encoded_list[i]
        return (
            torch.tensor(scaff, dtype=torch.long),
            torch.tensor(dec, dtype=torch.long),
        )  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @staticmethod
    def collate_fn(encoded_pairs):
        """
        Turns a list of encoded pairs (scaffold, decoration) of sequences and turns them into two batches.
        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the scaffolds and one for the decorations in the same order as given.
        """
        encoded_scaffolds, encoded_decorations = list(zip(*encoded_pairs))
        return (pad_batch(encoded_scaffolds), pad_batch(encoded_decorations))


def pad_batch(encoded_seqs):
    """
    Pads a batch.
    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """
    seq_lengths = torch.tensor(
        [len(seq) for seq in encoded_seqs], dtype=torch.int64
    )  # pylint: disable=not-callable
    return (tnnur.pad_sequence(encoded_seqs, batch_first=True), seq_lengths)
