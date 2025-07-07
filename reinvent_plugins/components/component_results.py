"""Dataclasses to hold transformed and aggregated result from components."""
from __future__ import annotations

__all__ = ["ComponentResults", "SmilesAssociatedComponentResults"]
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

import numpy as np


@dataclass
class SmilesResult:
    """Container for scores, uncertainty, failure data and metadata for a single smiles"""

    score: tuple  # multiple scores per component, and can be numbers or strings (as in MMP)
    score_property: Optional[List[Dict]] = None
    uncertainities: Optional[float] = None
    metadata: Optional[Dict] = None
    failures_properties: Optional[Dict] = None


@dataclass
class ComponentResults:
    """Container for the scores, uncertainties and metadata

    At the minimum the scores must be provided.  The order of the score array
    must be the same as the order of SMILES passed to each component.  Failure
    of computation of score must be indicated by NaN. Do not use zero for this!
    scores_properties can be used to pass on metadata on the scores
    uncertainty_type is currently assumed to be the same for all values
    failure_properties can be used to provide details on the failure of a component
    meta_data is a general facility to pass on metadata
    """

    scores: List[np.ndarray]
    scores_properties: Optional[List[Dict]] = None
    uncertainty: Optional[List[np.ndarray]] = None
    uncertainty_type: Optional[str] = None
    uncertainty_properties: Optional[List[Dict]] = None
    failures_properties: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


@dataclass
class SmilesAssociatedComponentResults:
    """Container for the scores, uncertainties and metadata, indexed by smiles.

    This class is built on top of ComponentResults to create an object with smiles-based indexing
    At the minimum the scores must be provided.
    Failure of computation of score must be indicated by NaN. Do not use zero for this!
    uncertainty_type is currently assumed to be the same for all values
    failure_properties can be used to provide details on the failure of a component
    meta_data is a general facility to pass on metadata per smiles


    """

    # this is the main holder for smiles-associated results
    data: Dict[str, smilesResult] = field(default=None)

    # this the base ComponentResults input
    component_results: Optional[ComponentResults] = field(init=False, default=None)

    # uncertainty type could be set a component level
    uncertainty_type: Optional[str] = field(default=None)

    @staticmethod
    def _score_lists_to_dict(
        smiles: List[str],
        scores: List[List[float | str]],
        metadata: Optional[Dict[str, List[float | str]]] = None,
    ) -> dict[str, smilesResult]:
        """Utility method for converting from scores in the form of [[score0_mol0,score0_mol1,...],
                                                                      [score1_mol0,score1_mol2,...]]]

        and possibly metadata of the form Dict[metadata0_name:[metadata0_mol0,metadata0_mol1,metadata0_mol2,...]]
        as returned by most scoring functions to dictionaries {smiles:SmilesResult((score_1,score2)}
        """
        values_to_upack = [smiles, list(zip(*scores))]

        if metadata is not None:
            # if we have metadata, we need to arrange it by smiles for each key.

            values_to_upack.append(
                [
                    dict(zip(list(metadata.keys()), value_group))
                    for value_group in list(zip(*metadata.values()))
                ]
            )
        else:
            # Metadata is optional so we need an empty placeholder
            values_to_upack.append([dict()] * len(smiles))

        return {
            smiles: SmilesResult(score=score, metadata=metadata)
            for smiles, score, metadata in zip(*values_to_upack)
        }

    def get_metadata_names(self):
        """
        utility function to collect all metadata tags by walking all dictionaries
        """
        metadata_names = set()
        for smile in self.data.keys():
            metadata_dictionary = self.data[smile].metadata
            for key in metadata_dictionary.keys():
                metadata_names.add(key)
        return metadata_names

    def __init__(
        self, component_results: ComponentResults = None, smiles: List[Str] = None, data=None
    ):

        """Constructor from either ComponentResults and smiles,  or setting th data directly"""
        if data is None:
            self.data = SmilesAssociatedComponentResults._score_lists_to_dict(
                smiles, component_results.scores, component_results.metadata
            )
        else:
            self.data = data

    @classmethod
    def create_from_scores(
        cls, smiles: list[str], scores: List[List]
    ) -> SmilesAssociatedComponentResults:
        """Method to create ComponentResults from a list of list of scores, as in components with multiple endpoints"""
        return cls([], data=SmilesAssociatedComponentResults._score_lists_to_dict(smiles, scores))

    def update_scores(self, smiles: List[str], scores: List[List[float | str]]) -> None:
        """Method to update ComponentResults object from a list of list of scores, as in components with multiple endpoints"""
        self.data.update(SmilesAssociatedComponentResults._score_lists_to_dict(smiles, scores))

    def fetch_scores(self, smiles: List[str], transpose=False) -> List:
        """Method to retrive a list of scores for a given list of smiles

        :param smiles: list of SMILES
        :param transpose: bool, if False dimensions are SMILES x endpoints, if True dimensions are endpoints x SMILES
        :returns: list of list of scores

        """
        # note that the input smiles are needed to ensure the list has a consistent order with validity masked in scorer
        scores_list = [self[smiles].score for smiles in smiles]  # first dimension is smiles

        if transpose:  # original csv writing and reporting functions need the data in this format
            scores_list = list(zip(*scores_list))  # first dimension is endpoints

        return scores_list

    def fetch_metadata(self, smiles: List[str], transpose=False) -> List:
        """Method to retrieve a dictionary of available metadata for a given list of smiles

        :param smiles: list of SMILES
        :param transpose: bool, if False dimensions are SMILES x endpoints, if True dimensions are endpoints x SMILES
        :returns: dictionary with metadata. Keys are metadata names, values are list of length SMILES

        """
        # first, update all metadata names available
        metadata_names = self.get_metadata_names()
        metadata_collection = dict()
        for metadata_name in metadata_names:
            metadata_collection[metadata_name] = []
            for smile in smiles:
                smile_metadata = self[smile].metadata
                if metadata_name in smile_metadata.keys():
                    metadata_collection[metadata_name].append(str(smile_metadata[metadata_name]))
                else:
                    metadata_collection[metadata_name].append(None)

        return metadata_collection

    def __getitem__(self, smiles: str) -> SmilesResult:
        """Retrieve all results data for a single smile strings"""
        return self.data[smiles]
