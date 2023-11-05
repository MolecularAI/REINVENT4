import unittest

import pytest

from reinvent.chemistry.library_design.aizynth.aizynth_client import AiZynthClient
from reinvent.chemistry.library_design.aizynth.synthetic_pathway_dto import (
    SyntheticPathwayDTO,
)
from tests.chemistry.fixtures.paths import (
    AIZYNTH_PREDICTION_URL,
    AIZYNTH_BUILDING_BLOCKS_URL,
    AIZYNTH_TOKEN,
)
from tests.chemistry.fixtures.test_data import COCAINE, CAFFEINE


class MockedLogger:
    def log_message(self, message):
        print(message)


@pytest.mark.skip(reason="requires a URL so not a unit test")
@pytest.mark.integration
class TestAiZynthClient(unittest.TestCase):
    def setUp(self):
        logger = MockedLogger()
        prediction_url = AIZYNTH_PREDICTION_URL
        availability_url = AIZYNTH_BUILDING_BLOCKS_URL
        api_token = AIZYNTH_TOKEN
        self._client = AiZynthClient(prediction_url, availability_url, api_token, logger)
        self._query_pathway = SyntheticPathwayDTO(precursors=[COCAINE])
        self._defective_query = SyntheticPathwayDTO(precursors=["123124#@$"])

    def test_synthesis_prediction(self):
        result = self._client.synthesis_prediction(CAFFEINE)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.pathways), 5)

    def test_get_stock_availability_1(self):
        result = self._client.get_stock_availability(COCAINE)
        self.assertFalse(result)

    def test_get_stock_availability_2(self):
        result = self._client.get_stock_availability(CAFFEINE)
        self.assertTrue(result)

    def test_pathway_stock_availability_score(self):
        pathway = self._client.synthesis_prediction(COCAINE)
        result = self._client.pathway_stock_availability_score(pathway)
        self.assertEqual(result, 0.5)

    def test_availability_score(self):
        result = self._client.availability_score(self._query_pathway)
        self.assertEqual(result, 0.0)

    def test_defective_query(self):
        result = self._client.availability_score(self._defective_query)
        self.assertEqual(result, 0.0)

    # def test_get_leaving_group_pairs_missing(self):
    #     with self.assertRaises(IOError) as context:
    #         leaving_groups = self._definitions.get_leaving_group_pairs('reductiveamination_1')
    #     self.assertTrue("there are no definitions for reaction name: reductiveamination_1" in str(context.exception))
