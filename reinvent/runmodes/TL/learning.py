"""Transfer learning (TL)

Train a given model with new data.  The data comes from a file with SMILES
strings.  The file is assumed to be in multi-column format separated by commas
(CSV) or spaces.  The SMILES string is extracted from the first column.

The SMILES strings is expected to be a complete molecule.
"""

from __future__ import annotations

__all__ = ["Learning"]
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect
from rdkit import Chem, DataStructs

from reinvent.runmodes.reporter.remote import get_reporter
from .reports.tensorboard import write_report, TBData
from .reports.remote import send_report, RemoteData


if TYPE_CHECKING:
    from reinvent.models import ModelAdapter
    from reinvent.runmodes.TL.configurations import Configuration

logger = logging.getLogger(__name__)


class Learning(ABC):
    """Trains a given model with new data from SMILES."""

    def __init__(
        self,
        model: ModelAdapter,
        tb_logdir: str,
        configuration: Configuration,
        logger_parameters,
    ):
        """Setup

        :param model: model adapter
        :param tb_logdir: name of the TensorBoard log directory
        :param configuration: configuration
        :param logger_parameters: parameters for remote logging
        """

        self.model = model
        self._config = configuration
        self.device = model.device

        try:
            _ = getattr(self.model, "sample_smiles")
            self.can_do_similarity = True
        except AttributeError:
            self.can_do_similarity = False

        # FIXME: SMILES standardization in preprocessing

        self._optimizer = configuration.optimizer
        self._lr_scheduler = configuration.learning_rate_scheduler

        self.ref_fps = None

        self.smilies = self._config.smilies
        self.validation_smilies = self._config.validation_smilies

        # FIXME: think what to do for Lib and LinkInvent
        if self.can_do_similarity:
            nmols = min(len(self.smilies), self._config.num_refs)
            ref_smilies = random.sample(self.smilies, nmols)
            mols = filter(
                lambda m: m, [Chem.MolFromSmiles(smiles) for smiles in ref_smilies]
            )
            self.ref_fps = [Chem.RDKFingerprint(mol) for mol in mols]

        self.sample_batch_size = max(self._config.sample_batch_size, 128)
        self.batch_size = configuration.batch_size
        self.save_freq = max(self._config.save_every_n_epochs, 1)

        self.reporter = get_reporter()
        self.tb_reporter = None
        if tb_logdir:
            self.tb_reporter = SummaryWriter(log_dir=tb_logdir)

            if self.can_do_similarity:
                mols = filter(
                    lambda m: m,
                    [Chem.MolFromSmiles(smiles) for smiles in self.smilies],
                )
                fps = [Chem.RDKFingerprint(mol) for mol in mols]

                sim = []

                for n in range(len(fps) - 1):
                    s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1 :])
                    sim.extend(s)

                self.tb_reporter.add_histogram(
                    "Tanimoto input SMILES", np.array(sim), 0
                )

        # FIXME: this is only available for Mol2mol
        if self._config.max_sequence_length:
            self.model.set_max_sequence_length(self._config.max_sequence_length)

    @abstractmethod
    def prepare_dataloader(self):
        """Prepare a pytorch Dataloader"""
        ...

    def _common_dataloader(self):
        """For Reinvent, LibInvent, LinkInvent"""

        # FIXME: had to set the generator explicitly to cuda to make this work
        #        shuffle=False would work do but probably not a good idea because
        #        it would affect training
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            generator=torch.Generator(device=self.device),
            shuffle=self._config.shuffle_each_epoch,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        self.validation_dataloader = None

        if self.validation_dataset:
            self.validation_dataloader = DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                generator=torch.Generator(device=self.device),
                shuffle=False,
                collate_fn=self.collate_fn,
                drop_last=False,
            )

    def optimize(self):
        start_epoch = self._config.starting_epoch - 1  # user input is 1 based
        end_epoch = start_epoch + self._config.num_epochs
        pbar = tqdm.tqdm(
            range(start_epoch, end_epoch),
            bar_format="{desc}|{bar}|{elapsed}",
            ascii=True,
        )

        self.prepare_dataloader()

        with tqdm_logging_redirect(loggers=[logger]):
            validation_losses = {}
            for epoch in pbar:
                self.model.set_mode("training")

                epoch_no = epoch + 1
                pbar.set_description(f"Epoch {epoch_no}")

                mean_nll = self.train_epoch()

                if epoch_no % self.save_freq == 0 or epoch_no == end_epoch:
                    mean_nll_valid = None
                    if self.validation_dataloader:
                        self.model.set_mode("inference")
                        stats = self.compute_stats(self.validation_dataloader)
                        mean_nll_valid = stats["nll"]
                        validation_losses[epoch_no] = mean_nll_valid

                    self.report(mean_nll, mean_nll_valid, epoch_no, end_epoch)
                    self._save_model(epoch_no)

                if self._terminate():
                    break

            self._save_model()

        if self.validation_dataloader:
            best_epoch_no = min(validation_losses, key=validation_losses.get)
            best_validation_loss = validation_losses[best_epoch_no]
            logger.info(
                f"Best validation loss ({best_validation_loss:.3f}) was at epoch {best_epoch_no:d}"
            )
            if best_epoch_no == max(validation_losses):
                logger.warning(
                    f"Best validation loss occured at the last epoch. Consider to train your model for more epochs"
                )

    @abstractmethod
    def train_epoch(self):
        ...

    @abstractmethod
    def compute_nll(self, batch):
        ...

    def _train_epoch_common(self) -> float:
        """Run one epoch of training

        :returns: mean negative log likelihood over all SMILES
        """

        mean_epoch_nlls = np.zeros(self.batch_size)

        for step, batch in enumerate(self.dataloader):
            nll = self.compute_nll(batch)
            loss = nll.mean()
            mean_epoch_nlls[step] = loss.item()

            self._optimizer.zero_grad()
            loss.backward()

            if self._config.clip_gradient_norm > 0:
                clip_grad_norm_(
                    self.model.network.parameters(), self._config.clip_gradient_norm
                )
            self._optimizer.step()

        self._lr_scheduler.step()  # Mol2Mol does this once per batch

        return mean_epoch_nlls.mean()

    def _terminate(self):
        terminate = False

        # FIXME: why are two steps made?
        self._lr_scheduler.step()

        new_lr = self._lr_scheduler.optimizer.param_groups[0]["lr"]

        if new_lr < self._config.learning_rate_config.min:
            terminate = True

        return terminate

    def _save_model(self, epoch: int = None) -> None:
        """Save the model to a file

        :param epoch: number when give to use for filename
        """

        suffix = f".{epoch}.chkpt" if epoch else ""
        path = f"{self._config.output_model_file}{suffix}"

        self.model.save_to_file(path)

    def compute_stats(self, dataloader: DataLoader) -> dict:
        """Compute several evaluation stats
        (only NLL is supported at the moment)

        :param dataloader: pytorch DataLoader object containing all the smilies
                           to use for evaluation
        """
        total_nll = 0.0
        n_examples = 0
        for step, batch in enumerate(dataloader):
            nll = self.compute_nll(batch).mean() * len(batch)
            total_nll = total_nll + nll.item()
            n_examples = n_examples + len(batch)
        return {"nll": total_nll / n_examples}

    def report(self, mean_nll: float, mean_nll_valid: float, epoch_no: int, epochs: int):
        """Log the report to various sources"""

        if self.tb_reporter:
            tb_data = TBData(
                epoch=epoch_no,
                mean_nll=mean_nll,
                mean_nll_valid=mean_nll_valid,
                sample_batch_size=self.sample_batch_size,
                ref_fps=self.ref_fps,
            )

            write_report(
                self.tb_reporter,
                tb_data,
                self.model,
                self.can_do_similarity,
                self.dataloader,
            )

        remote_data = RemoteData(
            epoch=epoch_no,
            epochs=epochs,
            mean_nll=mean_nll,
            mean_nll_valid=mean_nll_valid,
        )

        send_report(remote_data, self.reporter)
