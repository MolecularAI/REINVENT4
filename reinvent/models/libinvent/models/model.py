"""
Model class.
"""

from typing import List, Tuple

import torch
import torch.nn as tnn

from reinvent.models import meta_data
from reinvent.models.model_mode_enum import ModelModeEnum
from reinvent.models.libinvent.models.decorator import Decorator


class DecoratorModel:
    _model_type = "Libinvent"
    _version = 1

    def __init__(
        self,
        vocabulary,
        decorator,
        meta_data: meta_data.ModelMetaData,
        max_sequence_length=256,
        mode=ModelModeEnum.TRAINING,
        device=torch.device("cpu"),
    ):
        """
        Implements the likelihood and scaffold_decorating functions of the decorator model.
        :param vocabulary: A DecoratorVocabulary instance with the vocabularies of both the encoder and decoder.
        :param decorator: An decorator network instance.
        :param meta_data: model meta data
        :param max_sequence_length: Maximium number of tokens allowed to sample.
        :param mode: Mode in which the model should be initialized.
        :return:
        """

        self.vocabulary = vocabulary
        self.network = decorator
        self.network.to(device)
        self.meta_data = meta_data
        self.max_sequence_length = max_sequence_length
        self.device = device

        self._model_modes = ModelModeEnum()
        self.set_mode(mode)

        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

    @classmethod
    def load_from_file(cls, file_path: str, mode: str, device: torch.device):
        """
        Loads a model from a single file
        :param file_path: Path to the saved model.
        :return: An instance of the RNN.
        """

        save_dict = torch.load(file_path, map_location=device, weights_only=False)
        return cls.create_from_dict(save_dict, mode, device)

    @classmethod
    def create_from_dict(cls, save_dict: dict, mode: str, device: torch.device):
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        decorator = Decorator(**save_dict["decorator"]["params"])
        decorator.load_state_dict(save_dict["decorator"]["state"])

        model = cls(
            decorator=decorator,
            meta_data=save_dict["metadata"],
            mode=mode,
            device=device,
            **save_dict["model"],
        )

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            model=dict(
                vocabulary=self.vocabulary,
                max_sequence_length=self.max_sequence_length,
            ),
            decorator=dict(
                params=self.network.get_params(),
                state=self.network.state_dict(),
            ),
        )

        return save_dict

    def save(self, path):
        """
        Saves the model to a file.
        :param path: Path to the file which the model will be saved to.
        """

        save_dict = self.get_save_dict()

        torch.save(save_dict, path)

    save_to_file = save  # alias for backwards compatibility

    def set_mode(self, mode):
        """
        Changes the mode of the RNN to training or eval.
        :param mode: Mode to change to (training, eval)
        :return: The model instance.
        """
        if mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            self.network.train()
        return self

    def likelihood(
        self,
        scaffold_seqs,
        scaffold_seq_lengths,
        decoration_seqs,
        decoration_seq_lengths,
    ):
        """
        Retrieves the likelihood of a scaffold and its respective decorations.
        :param scaffold_seqs: (batch, seq) A batch of padded scaffold sequences.
        :param scaffold_seq_lengths: The length of the scaffold sequences (for packing purposes).
        :param decoration_seqs: (batch, seq) A batch of decorator sequences.
        :param decoration_seq_lengths: The length of the decorator sequences (for packing purposes).
        :return:  (batch) Log likelihood for each item in the batch.
        """

        # NOTE: the decoration_seq_lengths have a - 1 to prevent the end token to be forward-passed.
        logits = self.network(
            scaffold_seqs,
            scaffold_seq_lengths,
            decoration_seqs,
            decoration_seq_lengths - 1,
        )  # (batch, seq - 1, voc)
        log_probs = logits.log_softmax(dim=2).transpose(1, 2)  # (batch, voc, seq - 1)
        return self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(dim=1)  # (batch)

    @torch.no_grad()
    def sample_decorations(
        self, scaffold_seqs, scaffold_seq_lengths
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Samples as many decorations as scaffolds in the tensor.
        :param scaffold_seqs: a tensor with the scaffolds to sample already encoded and padded.
        :param scaffold_seq_lengths: a tensor with the length of the scaffolds.
        :return: a generator with (scaffold_smi, decoration_smi, nll) triplets.
        """
        batch_size = scaffold_seqs.size(0)

        input_vector = torch.full(
            (batch_size, 1),
            self.vocabulary.decoration_vocabulary["^"],
            dtype=torch.long,
        )  # (batch, 1)

        seq_lengths = torch.ones(batch_size)  # (batch)
        encoder_padded_seqs, hidden_states = self.network.forward_encoder(
            scaffold_seqs, scaffold_seq_lengths
        )

        nlls = torch.zeros(batch_size)
        not_finished = torch.ones(batch_size, 1, dtype=torch.long)
        sequences = []

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_states, _ = self.network.forward_decoder(
                input_vector, seq_lengths, encoder_padded_seqs, hidden_states
            )  # (batch, 1, voc)

            probs = logits.softmax(dim=2).squeeze()  # (batch, voc)
            log_probs = logits.log_softmax(dim=2).squeeze()  # (batch, voc)
            input_vector = torch.multinomial(probs, 1) * not_finished  # (batch, 1)
            sequences.append(input_vector)
            nlls += self._nll_loss(log_probs, input_vector.squeeze())
            not_finished = (input_vector > 1).type(torch.long)  # 0 is padding, 1 is end token

            if not_finished.sum() == 0:
                break

        decoration_smiles = [
            self.vocabulary.decode_decoration(seq)
            for seq in torch.cat(sequences, 1).data.cpu().numpy()
        ]

        scaffold_smiles = [
            self.vocabulary.decode_scaffold(seq) for seq in scaffold_seqs.data.cpu().numpy()
        ]

        return scaffold_smiles, decoration_smiles, nlls.data.cpu().numpy()

    def get_network_parameters(self):
        return self.network.parameters()
