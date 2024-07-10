"""Mol2Mol transfer learning"""

from __future__ import annotations

__all__ = ["Mol2Mol"]
import logging
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from .learning import Learning
import reinvent.models.transformer.core.dataset.paired_dataset as mol2mol_dataset

from ...models.transformer.mol2mol.dataset import get_pair_generator

logger = logging.getLogger(__name__)


class Mol2Mol(Learning):
    """Handle Mol2Mol specifics"""

    def prepare_dataloader(self):
        if self._config.ranking_loss_penalty and (self._config.pairs["type"] != "tanimoto"):
            raise ValueError("The ranking loss penalty is only supported for Tanimoto similarity")

        smilies = self.smilies

        if self.validation_smilies is None:
            validation_smilies = []
        else:
            validation_smilies = self.validation_smilies

        pairs = self._generate_pairs(smilies + validation_smilies)

        validation_mask = np.zeros(len(pairs), dtype=bool)

        for smile in validation_smilies:
            validation_mask |= pairs["Source_Mol"] == smile

        if validation_mask.sum() == len(pairs):
            raise ValueError("Smilies and validation smilies are the same set!")

        train_pairs = pairs.loc[~validation_mask].reset_index(drop=True)
        train_pairs.to_csv(
            os.path.join(os.path.dirname(self._config.output_model_file), "train.csv"),
            index=False,
        )
        logger.info("Indexing training pairs...")
        dataloader = self._create_dataloader(train_pairs)
        logger.info(f"Number of training pairs: {len(train_pairs):d}")

        validation_dataloader = None

        if validation_mask.sum() > 0:
            validation_pairs = pairs.loc[validation_mask].reset_index(drop=True)
            validation_pairs.to_csv(
                os.path.join(os.path.dirname(self._config.output_model_file), "valid.csv"),
                index=False,
            )
            logger.info("Indexing validation pairs...")
            validation_dataloader = self._create_dataloader(validation_pairs)
            logger.info(f"Number of validation pairs: {len(validation_pairs):d}")
        elif validation_mask.sum() == 0 and len(validation_smilies) > 0:
            logger.warning("No pairs were constructed from the validation set")

        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader

    def _generate_pairs(self, smilies):
        pair_config = self._config.pairs
        pair_generator = get_pair_generator(pair_config["type"], **pair_config)
        pairs = pair_generator.build_pairs(smilies, processes=self._config.n_cpus)

        if len(pairs) == 0:
            raise IOError("No valid entries are present in the supplied file")

        return pairs

    def _create_dataloader(self, data) -> DataLoader:
        """Initialize the dataloader.

        Needs too be updated for every epoch.

        :param data: data
        """
        args = dict(
            smiles_input=data["Source_Mol"],
            smiles_output=data["Target_Mol"],
            vocabulary=self.model.vocabulary,
            tokenizer=self.model.tokenizer,
            tanimoto_similarities=None if "Tanimoto" not in data else data["Tanimoto"],
        )

        dataset = mol2mol_dataset.PairedDataset(**args)
        collate_fn = mol2mol_dataset.PairedDataset.collate_fn
        drop_last = False

        dataloader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=self._config.shuffle_each_epoch,
            drop_last=drop_last,
            collate_fn=collate_fn,
            generator=torch.Generator(device=self.device),
        )
        return dataloader

    def train_epoch(self):
        total_loss = 0.0
        total_examples = 0.0
        use_ranking_loss = self._config.ranking_loss_penalty

        pbar = tqdm.tqdm(
            iterable=self.dataloader,
            total=len(self.dataloader),
            ascii=True,
            bar_format="{desc}|{bar}|{elapsed}",
        )

        for step, batch in enumerate(pbar):
            pbar.set_description(f"Batch {step}")

            src, src_mask, trg, trg_mask, sim = batch

            for x in src, src_mask, trg, trg_mask:
                x = x.to(self.device)

            self._optimizer.zero_grad()

            nll = self.compute_nll(batch)

            if use_ranking_loss:
                nll = torch.ravel(nll)
                sim = torch.ravel(sim.to(self.device))
                y = 2.0 * (sim[..., None] >= sim[None]) - 1
                ranking_loss = torch.maximum(torch.zeros_like(y), y * (nll[..., None] - nll[None]))
                ranking_loss = ranking_loss.sum() / (len(ranking_loss) * (len(ranking_loss) - 1))
                # FIXME: ranking loss lambda must be
                #        sent along with the other paramters
                loss = nll.mean() + 10.0 * ranking_loss.mean()
            else:
                loss = nll.mean()

            loss.backward()
            self._optimizer.step()  # must be called before LR scheduler step

            # FIXME: for the other generators this is done once per epoch
            self._lr_scheduler.step()

            total_examples += len(trg)
            total_loss += float(loss.detach().cpu().numpy()) * len(trg)

        loss_epoch = total_loss / total_examples

        return loss_epoch

    def compute_nll(self, batch):
        src, src_mask, trg, trg_mask, _ = batch

        return self.model.likelihood(src, src_mask, trg, trg_mask)
