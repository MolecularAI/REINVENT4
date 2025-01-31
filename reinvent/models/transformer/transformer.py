from abc import ABC
from typing import List, Tuple

import torch
from torch import nn as tnn
from torch.autograd import Variable

from reinvent.models import meta_data
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask
from reinvent.models.transformer.core.network.module.search import (
    beamsearch,
    Node,
    EOS,
    MaxLength,
    LogicalOr,
)
from reinvent.models.model_mode_enum import ModelModeEnum
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from reinvent.models.utils.consistify import consistify


class TransformerModel(ABC):
    def __init__(
        self,
        vocabulary: Vocabulary,
        network: EncoderDecoder,
        meta_data: meta_data.ModelMetaData,
        max_sequence_length: int = 128,
        mode: str = ModelModeEnum().TRAINING,
        device=torch.device("cpu"),
    ):
        self.vocabulary = vocabulary
        self.tokenizer = SMILESTokenizer()
        self.meta_data = meta_data

        self._model_modes = ModelModeEnum()
        self.network = network
        self.network.to(device)
        self.device = device
        self.set_mode(mode)

        self._sampling_modes_enum = SamplingModesEnum()
        self.max_sequence_length = max_sequence_length

        self.device = next(self.network.parameters()).device
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

        self.beam_size = 64

        # temperature: Factor by which the logits are divided.
        # Small numbers make the model more confident on each position, but also more conservative.
        # Large values result in more random predictions at each step.
        self.temperature = 1.0

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_temperature(self, temperature: float = 1.0):
        self.temperature = temperature

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, file_path: str, mode: str, device: torch.device):
        """
        Loads a model from a single file
        :param file_path: Path to the saved model
        :param mode: Mode in which the model should be initialized
        :return: An instance of the network
        """

        save_dict = torch.load(file_path, map_location=device, weights_only=False)
        return cls.create_from_dict(save_dict, mode, device)

    @classmethod
    def create_from_dict(cls, save_dict: dict, mode: str, device: torch.device):
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        network = EncoderDecoder(**save_dict["network_parameter"])
        network.load_state_dict(save_dict["network_state"])

        vocabulary = None
        if isinstance(save_dict["vocabulary"], dict):
            vocabulary = Vocabulary.load_from_dictionary(save_dict["vocabulary"])
        else:
            vocabulary = save_dict["vocabulary"]

        model = cls(
            vocabulary=vocabulary,
            network=network,
            meta_data=save_dict.get("metadata"),
            max_sequence_length=save_dict["max_sequence_length"],
            mode=mode,
            device=device,
        )

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary.get_dictionary(),
            max_sequence_length=self.max_sequence_length,
            network_parameter=self.network.get_params(),
            network_state=self.network.state_dict(),
        )

        return save_dict

    def save(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """

        save_dict = self.get_save_dict()

        torch.save(save_dict, path_to_file)

    save_to_file = save  # alias for backwards compatibility

    @consistify
    def likelihood(self, src, src_mask, trg, trg_mask):
        """
        Retrieves the likelihood of molecules.
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param trg: (batch, seq) A batch of output sequences; with start token, without end token.
        :param trg_mask: Mask of the input sequences.
        :return:  (batch) Log likelihood for each output sequence in the batch.
        """
        trg_y = trg[:, 1:]  # skip start token but keep end token
        trg = trg[:, :-1]  # save start token, skip end token
        out = self.network.forward(src, trg, src_mask, trg_mask)
        log_prob = self.network.generator(out, self.temperature).transpose(
            1, 2
        )  # (batch, voc, seq_len)
        nll = self._nll_loss(log_prob, trg_y).sum(dim=1)

        return nll

    @torch.no_grad()
    def sample(self, src, src_mask, decode_type) -> Tuple[List[str], List[str], List[float]]:
        """
        Sample molecules
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param decode_type: decode type
        """

        if not self._sampling_modes_enum.is_supported_sampling_mode(decode_type):
            raise ValueError(f"Sampling mode `{decode_type}` is not supported")

        if decode_type == self._sampling_modes_enum.BEAMSEARCH:
            beam_size = self.beam_size
            vocabulary = self.vocabulary
            tokenizer = self.tokenizer

            vocabulary.pad_token = 0  # 0 is padding
            vocabulary.bos_token = 1  # 1 is start symbol
            vocabulary.eos_token = 2  # 2 is end symbol

            node = Node(
                self.network,
                (src, src_mask),
                vocabulary,
                self.device,
                batch_size=128,  # internal batch_size used by beamsearch. 128 seems a good trade-off.
                data_device=self.device,
            )  # if it explodes use 'cpu' here
            stop_criterion = LogicalOr((MaxLength(self.max_sequence_length - 1), EOS()))

            beamsearch(node, beam_size, stop_criterion)

            input_smiles_list = []

            for seq in src.detach().cpu().numpy():
                s = tokenizer.untokenize(self.vocabulary.decode(seq))

                for _ in range(beam_size):
                    input_smiles_list.append(s)

            output_smiles_list = [
                tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in node.y.detach().cpu().numpy()
            ]

            nlls = (-node.loglikelihood.detach().cpu().numpy()).ravel()

        else:
            batch_size = src.shape[0]
            ys = torch.ones(1).to(self.device)
            ys = (
                ys.repeat(batch_size, 1).view(batch_size, 1).type_as(src.data)
            )  # shape [batch_size, 1]
            encoder_outputs = self.network.encode(src, src_mask)
            break_condition = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

            nlls = torch.zeros(batch_size).to(self.device)
            # FIXME: end_token = self.vocabulary.end_token
            end_token = self.vocabulary["$"]

            for i in range(self.max_sequence_length - 1):
                out = self.network.decode(
                    encoder_outputs,
                    src_mask,
                    Variable(ys),
                    Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
                )
                # (batch, seq, voc) need to exclude the probability of the start token "1"
                log_prob = self.network.generator(out[:, -1], self.temperature)
                prob = torch.exp(log_prob)

                mask_property_token = self.mask_property_tokens(batch_size)
                prob = prob.masked_fill(mask_property_token, 0)

                if decode_type == self._sampling_modes_enum.GREEDY:
                    _, next_word = torch.max(prob, dim=1)
                    # mask numbers after end token as 0
                    next_word = next_word.masked_fill(break_condition.to(self.device), 0)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]

                    # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                    nlls += self._nll_loss(log_prob, next_word)
                elif decode_type == self._sampling_modes_enum.MULTINOMIAL:
                    next_word = torch.multinomial(prob, 1)
                    # mask numbers after end token as 0
                    break_t = torch.unsqueeze(break_condition, 1).to(self.device)
                    next_word = next_word.masked_fill(break_t, 0)
                    ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                    next_word = torch.reshape(next_word, (next_word.shape[0],))

                    # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                    nlls += self._nll_loss(log_prob, next_word)

                # next_word = np.array(next_word.to('cpu').tolist())
                break_condition = break_condition | (next_word == end_token)

                if all(break_condition):  # end token
                    break

            tokenizer = SMILESTokenizer()
            input_smiles_list = [
                tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in src.detach().cpu().numpy()
            ]
            output_smiles_list = [
                tokenizer.untokenize(self.vocabulary.decode(seq))
                for seq in ys.detach().cpu().numpy()
            ]
            nlls = nlls.detach().cpu().numpy()

        return input_smiles_list, output_smiles_list, nlls

    def get_network_parameters(self):
        return self.network.parameters()

    def mask_property_tokens(self, batch_size):
        """
        Prevent model from sampling the property tokens even though it happens very rarely.
        When increasing temperature, it will increase the chance to sample property tokens.
        The ChEMBL prior in the paper below was trained with property tokens in the vocabulary.

        He, J., Nittinger, E., Tyrchan, C., Czechtizky, W., Patronov, A., Bjerrum, E. J., & Engkvist, O. (2022).
        Transformer-based molecular optimization beyond matched molecular pairs. Journal of cheminformatics, 14(1), 1-14.
        """
        property_tokens = [
            "LogD_(3.9, 4.1]",
            "LogD_(2.5, 2.7]",
            "LogD_(5.9, 6.1]",
            "LogD_(-6.1, -5.9]",
            "LogD_(3.3, 3.5]",
            "LogD_(-2.1, -1.9]",
            "LogD_(4.7, 4.9]",
            "LogD_(-4.5, -4.3]",
            "LogD_(0.7, 0.9]",
            "LogD_(-0.7, -0.5]",
            "LogD_(-4.7, -4.5]",
            "LogD_(-5.1, -4.9]",
            "LogD_(-6.5, -6.3]",
            "LogD_(3.5, 3.7]",
            "Solubility_no_change",
            "LogD_(-3.7, -3.5]",
            "LogD_(-1.9, -1.7]",
            "LogD_(-1.5, -1.3]",
            "LogD_(-0.3, -0.1]",
            "LogD_(6.7, 6.9]",
            "LogD_(-1.3, -1.1]",
            "LogD_(4.3, 4.5]",
            "Clint_no_change",
            "LogD_(0.3, 0.5]",
            "LogD_(-5.3, -5.1]",
            "LogD_(5.7, 5.9]",
            "LogD_(-0.9, -0.7]",
            "LogD_(5.3, 5.5]",
            "LogD_(6.9, inf]",
            "LogD_(-3.1, -2.9]",
            "LogD_(-3.9, -3.7]",
            "LogD_(5.5, 5.7]",
            "Clint_low->high",
            "LogD_(2.3, 2.5]",
            "LogD_(2.9, 3.1]",
            "LogD_(6.5, 6.7]",
            "LogD_(-2.7, -2.5]",
            "LogD_(-5.5, -5.3]",
            "LogD_(1.9, 2.1]",
            "LogD_(-3.5, -3.3]",
            "LogD_(-5.9, -5.7]",
            "LogD_(-6.3, -6.1]",
            "LogD_(-4.9, -4.7]",
            "LogD_(-3.3, -3.1]",
            "Solubility_high->low",
            "LogD_(-2.3, -2.1]",
            "LogD_(5.1, 5.3]",
            "LogD_(-0.1, 0.1]",
            "LogD_(3.1, 3.3]",
            "LogD_(-2.9, -2.7]",
            "LogD_(1.1, 1.3]",
            "LogD_(-2.5, -2.3]",
            "Clint_high->low",
            "LogD_(-1.1, -0.9]",
            "LogD_(4.5, 4.7]",
            "LogD_(-inf, -6.9]",
            "LogD_(6.3, 6.5]",
            "LogD_(-6.9, -6.7]",
            "LogD_(3.7, 3.9]",
            "LogD_(-4.1, -3.9]",
            "LogD_(1.7, 1.9]",
            "LogD_(2.7, 2.9]",
            "Solubility_low->high",
            "LogD_(4.9, 5.1]",
            "LogD_(4.1, 4.3]",
            "LogD_(-6.7, -6.5]",
            "LogD_(-1.7, -1.5]",
            "LogD_(0.1, 0.3]",
            "LogD_(-4.3, -4.1]",
            "LogD_(2.1, 2.3]",
            "LogD_(-0.5, -0.3]",
            "LogD_(0.9, 1.1]",
            "LogD_(6.1, 6.3]",
            "LogD_(0.5, 0.7]",
            "LogD_(-5.7, -5.5]",
            "LogD_(1.3, 1.5]",
            "LogD_(1.5, 1.7]",
        ]
        mask_property_token = torch.zeros(batch_size, len(self.vocabulary), dtype=torch.bool).to(
            self.device
        )
        for p_token in property_tokens:
            if p_token in self.vocabulary:
                i = self.vocabulary[p_token]
                mask_property_token[:, i] = True

        return mask_property_token
