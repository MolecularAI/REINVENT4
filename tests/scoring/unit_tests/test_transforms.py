import pytest

import numpy as np

from reinvent.scoring.transforms import DoubleSigmoid
from reinvent.scoring.transforms import ReverseSigmoid
from reinvent.scoring.transforms import Sigmoid
from reinvent.scoring.transforms import LeftStep
from reinvent.scoring.transforms import RightStep
from reinvent.scoring.transforms import Step
from reinvent.scoring.transforms import ValueMapping
from reinvent.scoring.transforms import ExponentialDecay


@pytest.mark.parametrize(
    "low, high, coef_div, coef_si, coef_se",
    [(-5.0, 5.0, 100.0, 150.0, 150.0), (4.0, 9.0, 35.0, 150.0, 150.0)],
)
def test_double_sigmoid(low, high, coef_div, coef_si, coef_se):
    from reinvent.scoring.transforms.double_sigmoid import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="", low=low, high=high, coef_div=coef_div, coef_si=coef_si, coef_se=coef_se
    )
    transform = DoubleSigmoid(params)
    results = transform(data)
    max_idx = np.argmax(results)
    assert np.all(results[1:max_idx] >= results[0 : max_idx - 1])
    assert np.all(results[max_idx:-1] >= results[max_idx + 1 :])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize("low, high, k", [(-5.0, 5.0, 1.0)])
def test_reverse_sigmoid(low, high, k):
    from reinvent.scoring.transforms.sigmoids import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="",
        low=low,
        high=high,
        k=k,
    )
    transform = ReverseSigmoid(params)
    results = transform(data)
    assert np.all(results[1:] <= results[:-1])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize("low, high, k", [(-5.0, 5.0, 1.0)])
def test_sigmoid(low, high, k):
    from reinvent.scoring.transforms.sigmoids import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="",
        low=low,
        high=high,
        k=k,
    )
    transform = Sigmoid(params)
    results = transform(data)
    assert np.all(results[1:] >= results[:-1])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize("mapping", [{0: 1, 1: 2, 2: 5}])
def test_value_mapping(mapping):
    from reinvent.scoring.transforms.value_mapping import Parameters

    data = [0, 0, 1, 2, 1, 2]
    params = Parameters(type="", mapping=mapping)
    transform = ValueMapping(params)
    results = transform(data)
    assert np.all(results == [1, 1, 2, 5, 2, 5])


@pytest.mark.parametrize("low, high", [(-5, 3)])
def test_left_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = LeftStep(params)
    results = transform(data)
    assert np.all(results == (data <= low))


@pytest.mark.parametrize("low, high", [(-5, 3)])
def test_right_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = RightStep(params)
    results = transform(data)
    assert np.all(results == (data >= high))


@pytest.mark.parametrize("low, high", [(-5, 3)])
def test_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = Step(params)
    results = transform(data)
    assert np.all(results == ((data <= high) & (data >= low)))


@pytest.mark.parametrize("k", [1.0, 10])
def test_exponential_decay(k):
    from reinvent.scoring.transforms.exponential_decay import Parameters

    data = np.linspace(-5, 5, 11, dtype=np.float32)
    params = Parameters(
        type="",
        k=k,
    )
    transform = ExponentialDecay(params)
    results = transform(data)
    assert np.all(np.diff(results) <= 0)  # Check that the function is monotonically decreasing.
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)
    assert results[0] == 1.0
    assert results[-1] <= 0.01
