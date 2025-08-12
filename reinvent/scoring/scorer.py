"""Prototype for the new heart of the scoring component

A scoring function is composed of components.  An aggregation function
combines components into the final scoring function.  Each component has a
weight and cna be transformed from a function.  Components report a primary
score for training and possibly secondary scores for reporting only.
Components also report uncertainties and failure of the actual scorer.
A scorer can be call through API, REST API or subprocess.
"""

from __future__ import annotations

__all__ = ["Scorer"]

from pathlib import Path
import multiprocessing as mp
from typing import List, Optional
import logging

import numpy as np

from reinvent.utils import config_parse
from . import aggregators
from .config import get_components
from .compute_scores import compute_transform
from .results import ScoreResults
from .validation import ScorerConfig

from pumas.aggregation import aggregation_catalogue


MAX_CPU_COUNT = 8
logger = logging.getLogger(__name__)


def setup_scoring(config: dict) -> dict:
    """Update scoring component from file if requested

    :param config: scoring dictionary
    :returns: scoring dictionary
    """

    component_filename = config.get("filename", "")
    component_filetype = config.get("filetype", "toml")

    if component_filename:
        component_filename = Path(component_filename).resolve()

        if component_filename.exists():
            ext = component_filename.suffix

            if ext in (f".{e}" for e in config_parse.INPUT_FORMAT_CHOICES):
                fmt = ext[1:]
            else:
                fmt = component_filetype

            logger.info(f"Reading score components from {component_filename}")

            input_config = config_parse.read_config(component_filename, fmt)
            config.update(input_config)
        else:
            logger.error(f"Component file {component_filename} not found")

    config["filename"] = None  # delete for dump as we now should have all components

    return config


def compute_component_score(component, fragments, smilies, valid_mask, use_pumas):
    fragments_component = component.component_type.startswith("fragment") or (
        component.component_type == "maize" and component.params[1].pass_fragments
    )
    if fragments and fragments_component:
        pass_smilies = fragments
    else:
        pass_smilies = smilies

    transform_result = compute_transform(
        component.component_type,
        component.params,
        pass_smilies,
        component.cache,
        valid_mask,
        index_smiles=smilies if fragments_component else None,
        use_pumas=use_pumas
    )

    return transform_result


class Scorer:
    """The main handler for a request to a scoring function"""

    def __init__(self, input_config: dict):
        """Set up the scorer

        :param input_config: scoring configuration
        """

        cfg = setup_scoring(input_config)
        config = ScorerConfig(**cfg)
        self.config = config

        self.use_pumas = config.use_pumas
        if self.use_pumas:
           
            # Need to handle mapping for back compatibility
            if config.type == 'custom_product':
                config.type = 'geometric_mean'
            if config.type == 'custom_sum':
                config.type = 'arithmetic_mean'

            self.aggregate = aggregation_catalogue.get(config.type)
        else:
            self.aggregate = getattr(aggregators, config.type)
        self.parallel = config.parallel

        self.components = get_components(config.component, use_pumas=self.use_pumas)

    def compute_results(
        self,
        smilies: List[str],
        invalid_mask: np.ndarray,
        duplicate_mask: np.ndarray,
        fragments: Optional[List[str]] = None,
        connectivity_annotated_smiles: Optional[List[str]] = None,
    ) -> ScoreResults:
        """Compute the score from a list of SMILES

        :param smilies: list of SMILES
        :param invalid_mask: mask for invalid SMILES
        :param duplicate_mask: mask for duplicate SMILES
        :param fragments: optional fragment SMILES
        :param connectivity_annotated_smiles: optional SMILES with added bonds annotated (LibInvent)
        :return: all results for the SMILES
        """

        # needs to be list for duplicate comps, name change for clearity
        completed_components = []

        ntasks = self.parallel

        valid_mask = np.logical_and(invalid_mask, duplicate_mask)
        filters_to_report, valid_mask = self.compute_filter_mask(
            smilies, valid_mask, connectivity_annotated_smiles
        )

        if ntasks > 1:
            nodes = min(MAX_CPU_COUNT, ntasks)

            ctx = mp.get_context("spawn")
            pool = ctx.Pool(nodes)

            number_components = (
                len(self.components.scorers)
                + len(self.components.filters)
                + len(self.components.penalties)
            )
            fragment_args = [fragments] * number_components
            smilies_args = [smilies] * number_components
            valid_mask_args = [valid_mask] * number_components

            completed_components = pool.starmap(
                compute_component_score,
                list(zip(self.components.scorers, fragment_args, smilies_args, valid_mask_args, self.use_pumas)),
            )
        else:
            for component in self.components.scorers:
                transform_result = compute_component_score(
                    component, fragments, smilies, valid_mask, self.use_pumas
                )
                completed_components.append(transform_result)

        scores_and_weights = []

        # PUMAS
        tscores_list = []
        weights_list = []

        for component in completed_components:
            for tscores, weight in zip(component.transformed_scores, component.weight):
                scores_and_weights.append((tscores, weight))

                if self.use_pumas:
                    tscores_list.append(tscores)
                    weights_list.append(weight)

        if len(scores_and_weights) > 0:  # penalty only run
            if self.use_pumas:
                total_scores = np.array([])
                for smiles in range(len(tscores_list[0])):
                    smiles_scores = [component[smiles] for component in tscores_list]
                    aggregate = self.aggregate()

                    if self.config.type == 'geometric_mean':
                        # Using the maxima and the float32 precision we get identical results to reinvent from the aggregation.
                        smiles_scores = np.maximum(np.array(smiles_scores, dtype=np.float32), 1e-8)
                        weights_list = np.maximum(np.array(weights_list, dtype=np.float32), 1e-8)
                    else:
                        # For other aggregation types, still use float32 as reinvent does.
                        smiles_scores = np.array(smiles_scores, dtype=np.float32)
                        weights_list = np.array(weights_list, dtype=np.float32) 

                    result = aggregate(values=smiles_scores, weights=weights_list)
                    total_scores = np.append(total_scores, result)
            else:
                total_scores = self.aggregate(scores_and_weights)
        else:
            total_scores = valid_mask.astype(float)  # apply filters if needed

        penalties = self.compute_penalties(completed_components, smilies, valid_mask)

        for filter_to_report in filters_to_report:
            completed_components.append(filter_to_report)

        return ScoreResults(smilies, total_scores * penalties, completed_components)

    def compute_penalties(self, completed_components, smilies, valid_mask):
        penalties = np.full(len(smilies), 1.0, dtype=float)

        for component in self.components.penalties:
            transform_result = compute_transform(
                component.component_type,
                component.params,
                smilies,
                component.cache,
                valid_mask,
                None,
                self.use_pumas
            )

            for scores in transform_result.transformed_scores:
                penalties *= scores

            completed_components.append(transform_result)

        return penalties

    def compute_filter_mask(self, smilies, valid_mask, connectivity_annotated_smiles):
        filters_to_report = []
        for component in self.components.filters:
            if connectivity_annotated_smiles and component.component_type == "reactionfilter":
                pass_smilies = connectivity_annotated_smiles
            else:
                pass_smilies = smilies

            transform_result = compute_transform(
                component.component_type,
                component.params,
                pass_smilies,
                component.cache,
                valid_mask,
                index_smiles=smilies if component.component_type == "reactionfilter" else None,
                use_pumas = self.use_pumas
            )

            for scores in transform_result.transformed_scores:
                valid_mask = np.logical_and(scores, valid_mask)
            # NOTE: filters are NOT also used as components as in REINVENT3

            filters_to_report.append(transform_result)

        return filters_to_report, valid_mask

    __call__ = compute_results
