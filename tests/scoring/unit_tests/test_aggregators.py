import numpy as np
import pytest

from reinvent.scoring.aggregators import arithmetic_mean
from reinvent.scoring.aggregators import geometric_mean


def test_arithmetic_mean():
    data1 = np.array([-4.43, -1.94, -2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, -7.40, 0.47, 2.29, -1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    result = arithmetic_mean(data)
    assert np.allclose(result, [-1.57, -4.86, -1.10, 5.17, 0.68], atol=1e-2)


def test_arithmetic_mean_nan():
    data1 = np.array([-4.43, -1.94, -2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, -7.40, np.nan, 2.29, -1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    result = arithmetic_mean(data)
    assert np.allclose(result, [-1.57, -4.86, -2.93, 5.17, 0.68], atol=1e-2)


def test_geometric_mean():
    data1 = np.array([4.43, 1.94, 2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, 7.40, 0.47, 2.29, 1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    result = geometric_mean(data)
    assert np.allclose(result, [1.88, 3.97, 1.09, 4.20, 2.62], atol=1e-2)


def test_geometric_mean_mismatch():
    data1 = np.array([4.43, 1.94, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, 7.40, 0.47, 2.29, 1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    with pytest.raises(ValueError):
        geometric_mean(data)


def test_geometric_mean_nan():
    data1 = np.array([4.43, 1.94, 2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, 7.40, np.nan, 2.29, 1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    result = geometric_mean(data)
    assert np.allclose(result, [1.88, 3.97, 2.93, 4.20, 2.62], atol=1e-2)


def test_geometric_mean_negative():
    data1 = np.array([4.43, 1.94, 2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, 7.40, -0.47, 2.29, 1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    result = geometric_mean(data)
    assert np.allclose(result, [1.88, 3.97, 0.0, 4.20, 2.62], atol=1e-2)
