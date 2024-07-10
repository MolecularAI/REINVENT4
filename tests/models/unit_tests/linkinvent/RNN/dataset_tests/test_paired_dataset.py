import unittest

from reinvent.models.linkinvent.dataset.paired_dataset import PairedDataset
from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)
from tests.test_data import (
    SCAFFOLD_SUZUKI,
    ETHANE,
    HEXANE,
    PROPANE,
    BUTANE,
)


class TestPairedDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles = [ETHANE, HEXANE, PROPANE, BUTANE]
        self.paired_model_voc = PairedModelVocabulary.from_lists(self.smiles, self.smiles)
        self.paired_data_set = PairedDataset(
            [list(i) for i in zip(self.smiles, self.smiles[::-1])],
            self.paired_model_voc,
        )
        (self.padded_warheads, self.warheads_seq_length), (
            self.padded_linker,
            self.linker_seq_length,
        ) = PairedDataset.collate_fn(self.paired_data_set)

    def test_len_both_valid(self):
        self.assertEqual(len(self.paired_data_set), 4)

    def test_coll_fn(self):
        self.assertEqual(len(self.padded_linker), len(self.smiles))
        self.assertEqual(len(self.padded_linker[0]), len(self.padded_linker[1]))
