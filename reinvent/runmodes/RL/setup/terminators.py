"""Callables to provide a termination criterion for Reinforcement Learning"""

from __future__ import annotations

__all__ = ["terminator_callable"]
import logging
from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial

logger = logging.getLogger(__name__)
terminator_callable = Callable  # generic as can be either function or class


def NullTerminator(a, b):
    """Do nothing terminator"""

    def do_nothing(c, d):
        return False

    return do_nothing


class SimpleTerminator:
    """Simply terminate on reaching a maximum score"""

    def __init__(self, max_score: float, min_steps: float):
        """Parameterise terminator.

        :param max_score: terminate above this score
        :param min_steps: minimum number of steps to carry out
        """
        self.max_score = max_score
        self.min_steps = min_steps

    def __call__(self, score: int, step: int) -> bool:
        """Terminate when score is larger than maximum

        Use average score for comparison.

        :param score: current score
        :param step: current step number
        """

        if step > self.min_steps and score >= self.max_score:
            return True

        return False


MAX_GRAD = 0.001  # FIXME: arbitray


class PlateauTerminator:
    """Terminate when a plateau is detected."""

    def __init__(self, max_score: float, min_steps: float, mem_size: int = 10):
        """Parameterise terminator.

        :param max_score: terminate above this score
        :param min_steps: minimum number of steps to carry out
        """
        self.max_score = max_score
        self.min_steps = min_steps
        self.mem_step = min_steps + mem_size

        self.memory = np.zeros(mem_size)
        self.x = np.arange(mem_size, dtype=float)
        self.mem_size = mem_size
        self.idx = 0

    def __call__(self, score: int, step: int) -> bool:
        """Terminate when score is larger than maximum

        Use average score for comparison.

        :param score: current score
        :param step: current step number
        """

        if step > self.min_steps:
            self.memory[self.idx] = score

            if self.idx < self.mem_size - 1:
                self.idx += 1
            else:
                self.idx = 0

            if step > self.mem_step:
                _, k = Polynomial.fit(self.x, self.memory, 1)

                logger.debug(f"Polyfit: {k=}")

                if abs(k) < MAX_GRAD:
                    return True

        return False


class TopkTerminator:
    """Terminate when the top-k scores of unique molecules no longer improves."""

    def __init__(self, patience: int, min_steps: float, topk: int = 10):
        """Parameterise terminator.

        :param patience: terminate if top-k stops improving for patience epochs
        :param min_steps: minimum number of steps to carry out
        :param topk: how many scores to consider
        """
        self.min_steps = min_steps
        self.topk = topk
        self.patience = patience
        self.count = 0
        self.sum = 0

        self.heap = []

    def __call__(self, scores: int, step: int) -> bool:
        """Terminate when top-k scores for unique SMILES stops improving

        :param scores: current scores
        :param step: current step number
        """

        for score in scores:
            if len(self.heap) < self.topk:
                heapq.heappush(self.heap, score)
            else:
                if score > self.heap[0]:
                    heapq.heapreplace(self.heap, score)

        if step > self.min_steps:
            new_sum = sum(self.heap)
            if new_sum > self.sum:
                self.sum = new_sum
                self.count = 0
            else:
                self.count += 1
            if self.count >= self.patience:
                return True

        return False
