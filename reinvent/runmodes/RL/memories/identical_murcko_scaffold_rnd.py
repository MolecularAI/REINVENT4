from typing import List, Optional, Tuple

import logging

from copy import deepcopy

import numpy as np
from reinvent.models import meta_data
import torch

from .diversity_filter import DiversityFilter

from reinvent.models.model_factory.sample_batch import SampleBatch

from reinvent import models

from reinvent.runmodes import create_adapter

logger = logging.getLogger(__name__)


class IdenticalMurckoScaffoldRND(DiversityFilter):
    """Provides intrinsic rewards based on random network distillation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        supported_novelty_functions = {
            "Reinvent": self._calculate_novelty_reinvent,
            "Libinvent": self._calculate_novelty_common,
            "Linkinvent": self._calculate_novelty_common,
            "Mol2Mol": self._calculate_novelty_common,
            "Pepinvent": self._calculate_novelty_common,
            "LinkinventTransformer": self._calculate_novelty_common,
            "LibinventTransformer": self._calculate_novelty_common,
        }

        self._prediction_network, _, model_type = create_adapter(
            dict_filename=self.prior_model_file_path,
            mode="training",
            device=self.device,
        )

        assert (
            model_type in supported_novelty_functions.keys()
        ), f"Model type {model_type} not supported for RND intrinsic reward. Supported types: {supported_novelty_functions.keys()}"

        self._add_intrinsic_rewards = supported_novelty_functions[model_type]

        # TODO: parameters could be shared except the last layer(s)
        # This would save memory, especially for the transformers model
        self._target_network = deepcopy(self._prediction_network)

        # Randomly initialize target network parameters and freeze them
        for param in self._target_network.get_network_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            param.requires_grad_(False)

        self._target_network.set_mode("inference")

        self._optimizer = torch.optim.Adam(
            self._prediction_network.get_network_parameters(), lr=self.learning_rate
        )

    def update_score(
        self,
        scores: np.ndarray,
        smilies: List[str],
        mask: np.ndarray,
        sampled: SampleBatch,
    ) -> Tuple[List, np.ndarray]:
        """Compute the score and add intrinsic rewards based on RND."""

        assert len(smilies) == len(
            sampled.items2
        ), f"Length of smilies ({len(smilies)}) and sampled.items2 ({len(sampled.items2)}) must be the same"

        scaffolds, original_scores, active_idxs = self.score_scaffolds(
            scores, smilies, mask, topological=False
        )

        self._add_intrinsic_rewards(sampled, scores, active_idxs)

        return scaffolds, original_scores

    def _calculate_novelty_reinvent(
        self, sampled: SampleBatch, scores: np.ndarray, active_idxs: List[int]
    ) -> np.ndarray:

        if len(active_idxs) == 0:
            return np.array([])

        smilies = [sampled.items2[i] for i in active_idxs]

        with torch.no_grad():
            target_likelihoods = self._target_network.likelihood_smiles(smilies)
        prediction_likelihoods = self._prediction_network.likelihood_smiles(smilies)

        loss = torch.pow(prediction_likelihoods - target_likelihoods, 2)

        novelty = loss.detach().cpu().numpy()

        # Backward propagate loss and perform one optimization step
        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        novelty = (novelty - np.amin(novelty)) / (np.amax(novelty) - np.amin(novelty) + 1e-6)

        for i, idx in enumerate(active_idxs):
            scores[idx] += novelty[i]

        return novelty

    def _calculate_novelty_common(
        self, sampled: SampleBatch, scores: np.ndarray, active_idxs: List[int]
    ) -> np.ndarray:

        if len(active_idxs) == 0:
            return np.array([])

        with torch.no_grad():
            target_likelihoods = self._target_network.likelihood_smiles(sampled).likelihood[
                active_idxs
            ]
        prediction_likelihoods = self._prediction_network.likelihood_smiles(sampled).likelihood[
            active_idxs
        ]

        loss = torch.pow(prediction_likelihoods - target_likelihoods, 2)

        novelty = loss.detach().cpu().numpy()

        # Backward propagate loss and perform one optimization step
        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        novelty = (novelty - np.amin(novelty)) / (np.amax(novelty) - np.amin(novelty) + 1e-6)

        for i, idx in enumerate(active_idxs):
            scores[idx] += novelty[i]

        return novelty
