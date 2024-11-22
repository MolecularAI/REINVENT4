"""Transfer learning (TL)

Train a given model with new data.  The data comes from a file with SMILES
strings.  The file is assumed to be in multi-column format separated by commas
(CSV) or spaces.  The SMILES string is extracted from the first column.

The SMILES strings is expected to be a complete molecule.
"""

from __future__ import annotations

__all__ = ["Learning"]
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from rdkit import Chem, DataStructs
import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from reinvent.runmodes.TL.reports import TLTBReporter, TLRemoteReporter, TLReportData
from reinvent.utils.logmon import get_reporter
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.utils.tensorboard import SummaryWriter  # monkey patch
from reinvent.models.meta_data import update_model_data
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.runmodes.utils import mutual_similarities, internal_diversity

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter
    from reinvent.runmodes.TL.validation import SectionParameters

logger = logging.getLogger(__name__)


class Learning(ABC):
    """Trains a given model with new data from SMILES."""

    def __init__(
        self,
        model: ModelAdapter,
        smilies: List[str],
        validation_smilies: List[str],
        tb_logdir: str,
        configuration: SectionParameters,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        lr_config,
    ):
        """Setup

        :param model: model adapter
        :param smilies: list of SMILES
        :param validation_smilies: list of validation SMILES
        :param tb_logdir: name of the TensorBoard log directory
        :param configuration: configuration
        :param optimizer: optimizer
        :param lr_scheduler: learning rate scheduler
        :param lr_config: learning rate configuration
        """

        self.model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_config = lr_config

        self._config = configuration
        self.device = model.device

        self.can_do_similarity = False
        self.validation_dataset = None
        self.collate_fn = None
        self.dataset = None
        self.randomize_all_smiles = self._config.randomize_all_smiles

        model_type = model.model_type

        # FIXME: ugly hard-coded model names
        if model_type == "Reinvent" or model_type == "Mol2Mol":
            self.can_do_similarity = True

        self.reference_fingerprints = None

        self.smilies = smilies
        self.validation_smilies = validation_smilies

        sample_batch_size = 1  # multiplier for sampled SMILES

        if model_type == "Reinvent":
            sample_batch_size = self._config.sample_batch_size

        sampling_parameters = {"batch_size": sample_batch_size}
        sampler, _ = setup_sampler(model_type, sampling_parameters, self.model)
        sampler.unique_sequences = False

        self.sampler = sampler
        self.sampling_smilies = random.choices(self.smilies, k=self._config.sample_batch_size)

        if not isinstance(self.sampling_smilies[0], str):
            self.sampling_smilies = [s[0] for s in self.sampling_smilies]

        do_similarity = self.can_do_similarity and self._config.num_refs > 0

        if do_similarity:
            self._prepare_similarity()

        self.batch_size = configuration.batch_size
        self.save_freq = max(self._config.save_every_n_epochs, 1)
        self.internal_diversity = self._config.internal_diversity

        self.reporters = []
        self._setup_reporters(do_similarity, model_type, tb_logdir)

        if hasattr(self._config, "max_sequence_length"):
            self.model.set_max_sequence_length(self._config.max_sequence_length)

        self.clip_gradient_norm = 0

        if hasattr(self._config, "clip_gradient_norm"):
            self.clip_gradient_norm = self._config.clip_gradient_norm

    @abstractmethod
    def prepare_dataloader(self):
        """Prepare a pytorch Dataloader"""
        ...

    def _common_dataloader(self):
        """For Reinvent, LibInvent, LinkInvent"""

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            generator=torch.Generator(device=self.device),
            shuffle=self._config.shuffle_each_epoch,
            collate_fn=self.collate_fn,
            drop_last=False,
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
            colour="green",
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

                    saved_model_path = self._save_model(epoch_no)

                    self.report(mean_nll, mean_nll_valid, epoch_no, saved_model_path)

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
                    f"No clear minimum found in validation loss. "
                    f"Consider training your model for additional epochs."
                )

    __call__ = optimize

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

        mean_epoch_nlls = np.zeros(len(self.dataloader))

        for step, batch in enumerate(self.dataloader):
            nll = self.compute_nll(batch)
            loss = nll.mean()
            mean_epoch_nlls[step] = loss.item()

            self._optimizer.zero_grad()
            loss.backward()

            if self.clip_gradient_norm > 0:
                clip_grad_norm_(self.model.network.parameters(), self.clip_gradient_norm)

            self._optimizer.step()

        self._lr_scheduler.step()  # Mol2Mol does this once per batch

        return mean_epoch_nlls.mean()

    def _terminate(self):
        terminate = False

        new_lr = self._lr_scheduler.optimizer.param_groups[0]["lr"]

        if new_lr < self._lr_config.min:
            terminate = True

        return terminate

    def _save_model(self, epoch: int = None) -> str:
        """Save the model to a file

        :param epoch: epoch number for checkpoint filename
        """

        suffix = f".{epoch}.chkpt" if epoch else ""
        path = f"{self._config.output_model_file}{suffix}"

        model_dict = self.model.get_save_dict()
        save_dict = update_model_data({**model_dict}, comment="TL")
        torch.save(save_dict, path)

        return path

    def _prepare_similarity(self):
        nmols = min(len(self.smilies), self._config.num_refs)
        ref_smilies = random.sample(self.smilies, nmols)
        mols = filter(lambda m: m, [Chem.MolFromSmiles(smiles) for smiles in ref_smilies])
        self.reference_fingerprints = [Chem.RDKFingerprint(mol) for mol in mols]

    def _compute_similarity(self, tb_reporter):
        mols = filter(
            lambda m: m,
            [Chem.MolFromSmiles(smiles) for smiles in self.smilies],
        )
        fps = [Chem.RDKFingerprint(mol) for mol in mols]

        sim = []

        for n in range(len(fps) - 1):
            s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1 :])
            sim.extend(s)

        tb_reporter.add_histogram("Tanimoto input SMILES", np.array(sim), 0)

    def compute_stats(self, dataloader: DataLoader) -> dict:
        """Compute several evaluation stats

        Only NLL is supported at the moment.

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

    def _setup_reporters(self, do_similarity, model_type, tb_logdir):
        """Setup for reporters"""

        remote_reporter = get_reporter()
        tb_reporter = None

        if tb_logdir:
            tb_reporter = SummaryWriter(log_dir=tb_logdir)

            # FIXME: allow this for all generators
            if model_type == "Reinvent":
                iv = torch.full((self.batch_size,), 0, dtype=torch.long).unsqueeze(1)
                tb_reporter.add_graph(self.model.network, input_to_model=iv)

            if do_similarity:
                self._compute_similarity(tb_reporter)

        # FIXME: needs a cleaner design, maybe move to caller
        for kls, reporter in (TLTBReporter, tb_reporter), (TLRemoteReporter, remote_reporter):
            if reporter:  # FIXME: remote reporter will be a class in any case
                self.reporters.append(kls(reporter))

    def report(
        self,
        mean_nll: float,
        mean_nll_valid: float,
        epoch_no: int,
        model_path: str,
    ):
        """Log the report to various sources"""

        samples = self.sampler.sample(self.sampling_smilies)
        sampled_smilies = []
        sampled_nlls = []
        mols = []
        duplicate_smiles = set()

        for smiles, nll, state in zip(samples.smilies, samples.nlls.cpu(), samples.states):
            if state == SmilesState.DUPLICATE:
                duplicate_smiles.add(smiles)

            if state == SmilesState.DUPLICATE or state == SmilesState.VALID:
                sampled_smilies.append(smiles)
                sampled_nlls.append(nll)

            mol = Chem.MolFromSmiles(smiles)

            if mol:
                mols.append(mol)

        intdiv = 0.0
        sampled_fps = [Chem.RDKFingerprint(mol) for mol in mols]

        if self.internal_diversity:
            similarities = mutual_similarities(sampled_fps)
            intdiv = internal_diversity(similarities, p=2)

        report_data = TLReportData(
            epoch=epoch_no,
            model_path=model_path,
            mean_nll=mean_nll,
            mean_nll_validation=mean_nll_valid,
            fingerprints=sampled_fps,
            reference_fingerprints=self.reference_fingerprints,
            sampled_smilies=sampled_smilies,
            sampled_nlls=np.array(sampled_nlls),
            fraction_valid=len(sampled_smilies) / len(samples.smilies),
            fraction_duplicates=len(duplicate_smiles) / len(samples.smilies),
            internal_diversity=intdiv,
        )

        for reporter in self.reporters:
            reporter.submit(report_data)
