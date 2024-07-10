import unittest

from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer
from tests.test_data import IBUPROFEN, NONSENSE


class TestSmilesTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SMILESTokenizer()

    def test_tokenize(self):
        self.assertListEqual(
            self.tokenizer.tokenize(IBUPROFEN),
            [
                "^",
                "C",
                "C",
                "(",
                "C",
                ")",
                "C",
                "c",
                "1",
                "c",
                "c",
                "c",
                "(",
                "c",
                "c",
                "1",
                ")",
                "[C@@H]",
                "(",
                "C",
                ")",
                "C",
                "(",
                "=",
                "O",
                ")",
                "O",
                "$",
            ],
        )

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
        self.assertEqual(
            self.tokenizer.untokenize(
                [
                    "^",
                    "C",
                    "C",
                    "(",
                    "C",
                    ")",
                    "C",
                    "c",
                    "1",
                    "c",
                    "c",
                    "c",
                    "(",
                    "c",
                    "c",
                    "1",
                    ")",
                    "[C@@H]",
                    "(",
                    "C",
                    ")",
                    "C",
                    "(",
                    "=",
                    "O",
                    ")",
                    "O",
                    "$",
                ]
            ),
            IBUPROFEN,
        )

        self.assertEqual(
            self.tokenizer.untokenize(
                ["C", "1", "C", "C", "(", "Br", ")", "C", "C", "C", "1", "[ClH]"]
            ),
            NONSENSE,
        )
