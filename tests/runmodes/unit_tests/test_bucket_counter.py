import pytest

from reinvent.runmodes.RL.memories import BucketCounter


@pytest.fixture
def bucket_counter():
    # no special setup
    yield BucketCounter(max_size=5, A=6, B=5, C=7, D=4)
    pass  # no tear-down


def test_bucket_full(bucket_counter):
    assert bucket_counter.bucket_full("A")
    assert bucket_counter.bucket_full("C")


@pytest.mark.xfail
def test_bucket_full_fail(bucket_counter):
    assert bucket_counter.bucket_full("B")
    assert bucket_counter.bucket_full("D")


def test_full_buckets(bucket_counter):
    assert tuple(bucket_counter.full_buckets()) == (("A", 6), ("C", 7))


def test_count_fill(bucket_counter):
    assert bucket_counter.count_full() == 2
