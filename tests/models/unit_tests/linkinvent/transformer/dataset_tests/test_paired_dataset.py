import unittest

import torch
import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from tests.models.unit_tests.linkinvent.transformer.fixtures import mocked_vocabulary
from tests.test_data import WARHEAD_PAIR, SCAFFOLD_TO_DECORATE, WARHEAD_TRIPLE, LINKER_TRIPLE


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.smiles_input = [WARHEAD_PAIR, WARHEAD_TRIPLE]
        self.smiles_output = [SCAFFOLD_TO_DECORATE, LINKER_TRIPLE]
        self.vocabulary = mocked_vocabulary()
        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, smiles_input, smiles_output):
        dataset = PairedDataset(
            smiles_input,
            smiles_output,
            vocabulary=self.vocabulary,
            tokenizer=SMILESTokenizer(),
        )
        dataloader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn
        )
        return dataloader

    def _get_src(self):
        for batch in self.data_loader:
            return batch.input

    def _get_src_mask(self):
        for batch in self.data_loader:
            return batch.input_mask

    def _get_trg(self):
        for batch in self.data_loader:
            return batch.output

    def _get_trg_mask(self):
        for batch in self.data_loader:
            return batch.output_mask

    def _get_src_shape(self):
        for batch in self.data_loader:
            return batch.input.shape

    def _get_src_mask_shape(self):
        for batch in self.data_loader:
            return batch.input_mask.shape

    def _get_trg_shape(self):
        for batch in self.data_loader:
            return batch.output.shape

    def _get_trg_mask_shape(self):
        for batch in self.data_loader:
            return batch.output_mask.shape

    def test_src_shape(self):
        result = self._get_src_shape()
        self.assertEqual(list(result), [2, 38])

    def test_src_mask_shape(self):
        result = self._get_src_mask_shape()
        self.assertEqual(list(result), [2, 1, 38])

    def test_trg_shape(self):
        result = self._get_trg_shape()
        self.assertEqual(list(result), [2, 58])

    def test_trg_mask_shape(self):
        result = self._get_trg_mask_shape()
        self.assertEqual(list(result), [2, 57, 57])

    def test_src_content(self):
        result = self._get_src()
        comparison = torch.equal(
            result,
            torch.tensor(
                [
                    [
                        1,
                        7,
                        13,
                        9,
                        13,
                        13,
                        13,
                        13,
                        13,
                        9,
                        22,
                        7,
                        13,
                        9,
                        13,
                        13,
                        13,
                        13,
                        5,
                        16,
                        15,
                        6,
                        13,
                        9,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        1,
                        7,
                        15,
                        5,
                        13,
                        6,
                        13,
                        22,
                        7,
                        13,
                        20,
                        9,
                        20,
                        21,
                        20,
                        20,
                        5,
                        13,
                        4,
                        15,
                        6,
                        20,
                        9,
                        22,
                        7,
                        13,
                        19,
                        5,
                        16,
                        6,
                        13,
                        13,
                        5,
                        12,
                        16,
                        6,
                        16,
                        2,
                    ],
                ]
            ),
        )
        self.assertTrue(comparison)

    def test_src_mask_content(self):
        result = self._get_src_mask()
        comparison = torch.equal(
            result,
            torch.tensor(
                [
                    [
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ]
                    ],
                    [
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ]
                    ],
                ]
            ),
        )
        self.assertTrue(comparison)

    def test_trg_content(self):
        result = self._get_trg()
        comparison = torch.equal(
            result,
            torch.tensor(
                [
                    [
                        1,
                        18,
                        20,
                        9,
                        20,
                        20,
                        20,
                        5,
                        20,
                        20,
                        9,
                        6,
                        20,
                        10,
                        20,
                        20,
                        5,
                        21,
                        21,
                        10,
                        20,
                        11,
                        20,
                        20,
                        20,
                        5,
                        20,
                        20,
                        11,
                        6,
                        17,
                        5,
                        12,
                        16,
                        6,
                        5,
                        12,
                        16,
                        6,
                        15,
                        6,
                        18,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        1,
                        18,
                        13,
                        20,
                        9,
                        20,
                        20,
                        20,
                        5,
                        8,
                        20,
                        10,
                        20,
                        20,
                        20,
                        20,
                        5,
                        13,
                        16,
                        20,
                        11,
                        20,
                        20,
                        5,
                        16,
                        18,
                        6,
                        20,
                        5,
                        13,
                        15,
                        18,
                        6,
                        20,
                        20,
                        11,
                        14,
                        6,
                        20,
                        10,
                        13,
                        6,
                        20,
                        5,
                        13,
                        6,
                        20,
                        9,
                        8,
                        20,
                        9,
                        20,
                        20,
                        20,
                        20,
                        20,
                        9,
                        2,
                    ],
                ]
            ),
        )
        self.assertTrue(comparison)
