import unittest

import torch
import torch.utils.data as tud

from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from tests.models.unit_tests.libinvent.transformer.fixtures import mocked_vocabulary
from tests.test_data import SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT, \
    SCAFFOLD_QUADRUPLE_POINT, DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.smiles_input = [SCAFFOLD_SINGLE_POINT, SCAFFOLD_DOUBLE_POINT, SCAFFOLD_TRIPLE_POINT,
                             SCAFFOLD_QUADRUPLE_POINT]
        self.smiles_output = [DECORATION_NO_SUZUKI, TWO_DECORATIONS_ONE_SUZUKI, THREE_DECORATIONS, FOUR_DECORATIONS]
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
        self.assertEqual(list(result), [4, 60])

    def test_src_mask_shape(self):
        result = self._get_src_mask_shape()
        self.assertEqual(list(result), [4, 1, 60])

    def test_trg_shape(self):
        result = self._get_trg_shape()
        self.assertEqual(list(result), [4, 27])

    def test_trg_mask_shape(self):
        result = self._get_trg_mask_shape()
        self.assertEqual(list(result), [4, 26, 26])

    def test_src_content(self):
        result = self._get_src()
        comparison = torch.equal(
            result,
            torch.tensor([[
                1, 15, 10, 4, 9, 13, 5, 12, 16, 4, 10, 5, 10, 4, 9, 13, 5, 10,
                4, 10, 5, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0],
                [1, 15, 10, 4, 10, 10, 19, 7, 19, 19, 18, 8, 18, 18, 18, 18, 18, 8,
                 18, 7, 9, 13, 5, 10, 4, 9, 13, 5, 15, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [1, 15, 18, 7, 19, 18, 4, 15, 5, 19, 18, 4, 12, 5, 18, 7, 10, 4,
                 9, 13, 5, 10, 16, 4, 10, 12, 14, 4, 12, 5, 4, 9, 13, 5, 9, 13,
                 5, 10, 4, 9, 13, 5, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [1, 15, 10, 4, 9, 13, 5, 10, 7, 10, 10, 4, 13, 5, 10, 12, 7, 10,
                 4, 13, 5, 10, 4, 17, 7, 18, 18, 4, 15, 5, 19, 4, 6, 18, 8, 18,
                 18, 18, 4, 15, 5, 18, 4, 11, 5, 18, 8, 5, 19, 7, 5, 10, 4, 10,
                 5, 4, 10, 5, 15, 2]]),
        )
        self.assertTrue(comparison)

    def test_src_mask_content(self):
        result = self._get_src_mask()
        comparison = torch.equal(
            result,
            torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False]],

                          [[True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            False, False, False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False]],

                          [[True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, False, False, False, False, False, False,
                            False, False, False, False, False, False, False, False, False, False]],

                          [[True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True, True]]]),
        )
        self.assertTrue(comparison)

    def test_trg_content(self):
        result = self._get_trg()
        comparison = torch.equal(
            result,
            torch.tensor([[1, 15, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 15, 18, 7, 19, 18, 19, 18, 18, 7, 20, 15, 10, 2, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 15, 18, 7, 19, 18, 19, 18, 18, 7, 20, 15, 18, 7, 19, 18, 19, 18,
                           18, 7, 20, 15, 10, 2, 0, 0, 0],
                          [1, 15, 18, 7, 19, 18, 19, 18, 18, 7, 20, 15, 18, 7, 19, 18, 19, 18,
                           18, 7, 20, 15, 10, 20, 15, 10, 2]]),
        )
        self.assertTrue(comparison)
