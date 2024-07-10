from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data as tud

from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)


class PairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    def __init__(self, input_target_smi_list: List[List[str]], vocabulary: PairedModelVocabulary):
        self.vocabulary = vocabulary
        self._encoded_list = []

        for input_smi, target_smi in input_target_smi_list:
            en_input = self.vocabulary.input.encode(input_smi)
            en_output = self.vocabulary.target.encode(target_smi)

            if en_input is not None and en_output is not None:
                self._encoded_list.append((en_input, en_output))
            else:
                pass
                # TODO log theses cases

    def __getitem__(self, i):
        en_input, en_output = self._encoded_list[i]
        return (
            torch.tensor(en_input, dtype=torch.long),
            torch.tensor(en_output, dtype=torch.long),
        )  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @staticmethod
    def collate_fn(encoded_pairs):
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """

        encoded_inputs, encoded_targets = list(zip(*encoded_pairs))

        return _pad_batch(encoded_inputs), _pad_batch(encoded_targets)


def _pad_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
    """Pads a batch.

    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """

    seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)

    return pad_sequence(encoded_seqs, batch_first=True), seq_lengths
