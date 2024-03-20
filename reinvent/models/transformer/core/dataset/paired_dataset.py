from typing import List, Tuple
import logging

from torch import Tensor
from torch.autograd import Variable
from torch.utils import data as tud
import numpy as np
import torch

from reinvent.models.transformer.core.dto.batch_dto import BatchDTO
from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask

DEVICE = "cpu"
logger = logging.getLogger(__name__)


class PairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""

    # TODO check None for en_input, en_output
    def __init__(
        self,
        smiles_input: List[str],
        smiles_output: List[str],
        vocabulary,
        tokenizer,
        tanimoto_similarities: List[float] = None,
    ):
        self.vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_input_list = []
        self._encoded_output_list = []
        self._tanimoto_similarities = []

        if tanimoto_similarities is None:
            tanimoto_similarities = np.zeros(len(smiles_input), dtype=np.float32)

        for input_smi, output_smi, tanimoto in zip(
            smiles_input, smiles_output, tanimoto_similarities
        ):
            ok_input, ok_output = True, True
            try:
                tokenized_input = self._tokenizer.tokenize(input_smi)
                en_input = self.vocabulary.encode(tokenized_input)
            except KeyError as e:
                logger.warning(
                    f"Input smile {input_smi} contains an invalid token {e}. It will be ignored"
                )
                ok_input = False
            try:
                tokenized_output = self._tokenizer.tokenize(output_smi)
                en_output = self.vocabulary.encode(tokenized_output)
            except KeyError as e:
                logger.warning(
                    f"Output smile {output_smi} contains an invalid token {e}. It will be ignored"
                )
                ok_output = False
            if ok_input and ok_output:
                self._encoded_input_list.append(en_input)
                self._encoded_output_list.append(en_output)
                self._tanimoto_similarities.append(tanimoto)

    def __getitem__(self, i):
        en_input, en_output = self._encoded_input_list[i], self._encoded_output_list[i]
        tanimoto = self._tanimoto_similarities[i]
        return (
            torch.tensor(en_input).long(),
            torch.tensor(en_output).long(),
            torch.tensor(tanimoto).float().view(1),
        )  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_input_list)

    @staticmethod
    def collate_fn(encoded_pairs) -> BatchDTO:
        """Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.

        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """

        encoded_inputs, encoded_targets, tanimoto_similarities = list(zip(*encoded_pairs))
        collated_arr_source, src_mask = _mask_batch(encoded_inputs)
        collated_arr_target, trg_mask = _mask_batch(encoded_targets)

        # TODO: refactor the logic below
        trg_mask = trg_mask & Variable(
            subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask)
        )
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = BatchDTO(
            collated_arr_source,
            src_mask,
            collated_arr_target,
            trg_mask,
            torch.cat(tanimoto_similarities),
        )
        return dto


def _mask_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
    """Pads a batch.

    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded and masked
    """

    # maximum length of input sequences
    max_length_source = max([seq.size(0) for seq in encoded_seqs])

    # padded source sequences with zeroes
    collated_arr_seq = torch.zeros(len(encoded_seqs), max_length_source).long()
    seq_mask = torch.zeros(len(encoded_seqs), 1, max_length_source).bool()

    for i, seq in enumerate(encoded_seqs):
        collated_arr_seq[i, : len(seq)] = seq
        seq_mask[i, 0, : len(seq)] = True

    # mask of source seqs
    # seq_mask = (collated_arr_seq != 0).unsqueeze(-2)

    return collated_arr_seq, seq_mask
