from typing import List

import requests

from reinvent.chemistry.library_design.aizynth.collection_of_pathways_dto import (
    CollectionOfPathwaysDTO,
)
from reinvent.chemistry.library_design.aizynth.synthetic_pathway_dto import SyntheticPathwayDTO


class AiZynthClient:
    """Currently this class is specific for the internal REST API of AstraZeneca"""

    def __init__(self, prediction_url: str, availability_url: str, api_token: str, logger):
        self._prediction_url = prediction_url
        self._availability_url = availability_url
        self._headers = self._compose_headers(api_token)
        self._logger = logger

    def synthesis_prediction(self, smile: str) -> CollectionOfPathwaysDTO:
        data = {"smiles": smile, "policy": "Full Set"}
        try:
            response = requests.post(self._prediction_url, headers=self._headers, data=data)
            response.raise_for_status()
            result = response.json()
            precursor_sets = [
                SyntheticPathwayDTO(precursors=precursor["smiles_split"])
                for precursor in result["precursors"]
            ]
            pathway_collection = CollectionOfPathwaysDTO(input=smile, pathways=precursor_sets)
            return pathway_collection
        except requests.exceptions.HTTPError as e:
            self._logger.log_message(e)
            self._logger.log_message(f"Failed for string: {smile}")
            return CollectionOfPathwaysDTO(input=smile, pathways=[])

    def batch_synthesis_prediction(self, smiles: List[str]) -> List[CollectionOfPathwaysDTO]:
        precursors = [self.synthesis_prediction(smile) for smile in smiles]
        return precursors

    def get_stock_availability(self, smile: str) -> bool:
        response = requests.get(self._availability_url, headers=self._headers, params={"q": smile})

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self._logger.log_message(e)

        if response.status_code == requests.codes.ok:
            result = response.json()
            return len(result["result"]) > 0
        return False

    def availability_score(self, pathway: SyntheticPathwayDTO) -> float:
        count = 0
        for building_block in pathway.precursors:
            if self.get_stock_availability(building_block):
                count += 1
        score = count / max(1, len(pathway.precursors))
        return score

    def pathway_stock_availability_score(self, pathways: CollectionOfPathwaysDTO):
        scores = [self.availability_score(pathway) for pathway in pathways.pathways]
        best_score = max(scores) if scores else 0
        return best_score

    def batch_stock_availability_score(
        self, pathways: List[CollectionOfPathwaysDTO]
    ) -> List[float]:
        scores = [self.pathway_stock_availability_score(pathway) for pathway in pathways]
        return scores

    def _compose_headers(self, api_token: str):
        headers = {"accept": "application/json", "Authorization": f"Token {api_token}"}
        return headers
