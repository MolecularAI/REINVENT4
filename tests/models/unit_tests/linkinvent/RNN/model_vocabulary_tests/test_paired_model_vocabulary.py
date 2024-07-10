import unittest

from reinvent.chemistry import tokens

from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)
from reinvent.models.linkinvent.model_vocabulary.vocabulary import (
    create_vocabulary,
    SMILESTokenizer,
)
from tests.test_data import WARHEAD_PAIR, SCAFFOLD_SUZUKI


class TestPairedModelVocabulary(unittest.TestCase):
    def setUp(self) -> None:
        warheads_list = [WARHEAD_PAIR]
        linker_list = [SCAFFOLD_SUZUKI]
        self.pmv = PairedModelVocabulary(
            create_vocabulary(warheads_list, SMILESTokenizer()),
            SMILESTokenizer(),
            create_vocabulary(linker_list, SMILESTokenizer()),
            SMILESTokenizer(),
        )

    def test_len(self):
        self.assertEqual(self.pmv.len(), (11, 16))

    def test_correct_tokens(self):
        warhead_sep = tokens.ATTACHMENT_SEPARATOR_TOKEN
        self.assertIsNotNone(self.pmv.input.encode(warhead_sep))