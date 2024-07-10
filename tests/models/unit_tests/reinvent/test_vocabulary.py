import unittest

import numpy as np
import numpy.testing as npt

from reinvent.models.reinvent.models.vocabulary import Vocabulary
from tests.test_data import SIMPLE_TOKENS


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.voc = Vocabulary(tokens=SIMPLE_TOKENS)

    def test_add_to_vocabulary_1(self):
        idx = self.voc.add("#")
        self.assertIn("#", self.voc)
        self.assertIn(idx, self.voc)
        self.assertEqual(self.voc["#"], idx)
        self.assertEqual(self.voc[idx], "#")

    def test_add_to_vocabulary_2(self):
        idx = self.voc.add("7")
        self.assertIn("7", self.voc)
        self.assertIn(idx, self.voc)
        self.assertEqual(self.voc["7"], idx)
        self.assertEqual(self.voc[idx], "7")

    def test_add_to_vocabulary_3(self):
        idx = self.voc.add("1")
        self.assertIn("1", self.voc)
        self.assertIn(idx, self.voc)
        self.assertEqual(self.voc[idx], "1")
        self.assertEqual(self.voc["1"], idx)

    def test_add_to_vocabulary_4(self):
        with self.assertRaises(TypeError) as context:
            self.voc.add(1)
        self.assertIn("Token is not a string", str(context.exception))

    def test_includes(self):
        self.assertIn(2, self.voc)
        self.assertIn("1", self.voc)
        self.assertNotIn(21, self.voc)
        self.assertNotIn("6", self.voc)

    def test_equal(self):
        self.assertEqual(self.voc, Vocabulary(tokens=SIMPLE_TOKENS))
        self.voc.add("#")
        self.assertNotEqual(self.voc, Vocabulary(tokens=SIMPLE_TOKENS))

    def test_update_vocabulary_1(self):
        idxs = self.voc.update(["5", "#"])
        self.assertIn("5", self.voc)
        self.assertIn(idxs[0], self.voc)
        self.assertIn("#", self.voc)
        self.assertIn(idxs[1], self.voc)
        self.assertEqual(self.voc["5"], idxs[0])
        self.assertEqual(self.voc[idxs[0]], "5")
        self.assertEqual(self.voc["#"], idxs[1])
        self.assertEqual(self.voc[idxs[1]], "#")

    def test_update_vocabulary_2(self):
        idx = self.voc.update(["1", "2"])
        self.assertIn("1", self.voc)
        self.assertIn("2", self.voc)
        self.assertIn(idx[0], self.voc)
        self.assertIn(idx[1], self.voc)
        self.assertEqual(self.voc["1"], idx[0])
        self.assertEqual(self.voc["2"], idx[1])
        self.assertEqual(idx[0], self.voc["1"])
        self.assertEqual(idx[1], self.voc["2"])
        self.assertEqual("1", self.voc[4])
        self.assertEqual("2", self.voc[5])
        self.assertEqual("1", self.voc[idx[0]])
        self.assertEqual("2", self.voc[idx[1]])

    def test_update_vocabulary_3(self):
        with self.assertRaises(TypeError) as context:
            self.voc.update([1, 2])
        self.assertIn("Token is not a string", str(context.exception))

    def test_delete_vocabulary_1(self):
        idx3 = self.voc["1"]
        del self.voc["1"]
        self.assertNotIn("1", self.voc)
        self.assertNotIn(idx3, self.voc)

    def test_delete_vocabulary_2(self):
        idx4 = self.voc[5]
        del self.voc[5]
        self.assertNotIn("2", self.voc)
        self.assertNotIn(idx4, self.voc)

    def test_len(self):
        self.assertEqual(len(self.voc), 15)
        self.assertEqual(len(Vocabulary()), 0)

    def test_encode(self):
        npt.assert_almost_equal(self.voc.encode(["^", "C", "C", "$"]), np.array([1, 8, 8, 0]))

    def test_decode(self):
        self.assertEqual(self.voc.decode(np.array([0, 8, 9, 8, 1])), ["$", "C", "F", "C", "^"])
