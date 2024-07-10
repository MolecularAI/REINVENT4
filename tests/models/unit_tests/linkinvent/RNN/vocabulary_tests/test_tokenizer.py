import unittest

from reinvent.models.linkinvent.model_vocabulary.model_vocabulary import SMILESTokenizer
from tests.test_data import IBUPROFEN, IBUPROFEN_TOKENIZED


class TestSmilesTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SMILESTokenizer()

    def test_tokenize(self):
        self.assertListEqual(self.tokenizer.tokenize(IBUPROFEN), IBUPROFEN_TOKENIZED)

        self.assertListEqual(
            self.tokenizer.tokenize("C%12CC(Br)C1CC%121[ClH]", with_begin_and_end=False),
            [
                "C",
                "%12",
                "C",
                "C",
                "(",
                "Br",
                ")",
                "C",
                "1",
                "C",
                "C",
                "%12",
                "1",
                "[ClH]",
            ],
        )

    def test_untokenize(self):
        self.assertEqual(self.tokenizer.untokenize(IBUPROFEN_TOKENIZED), IBUPROFEN)

        self.assertEqual(
            self.tokenizer.untokenize(
                ["C", "1", "C", "C", "(", "Br", ")", "C", "C", "C", "1", "[ClH]"]
            ),
            "C1CC(Br)CCC1[ClH]",
        )
