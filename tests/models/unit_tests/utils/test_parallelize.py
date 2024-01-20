from functools import partial
import os

import numpy as np

from reinvent.models.utils.parallel import parallel


def sigmoid(data, *, weights):
    for _ in range(1000):
        x = 1 / (1 + np.exp(-data * weights))
    return x


def test_parallel():
    data = np.random.randn(1000)
    w = 1.3

    res = sigmoid(data, weights=w)
    parallel_sigmoid = partial(parallel, os.cpu_count(), {"weights": w}, sigmoid)
    res_parallel = parallel_sigmoid(data)

    assert np.allclose(res, np.concatenate(res_parallel))
