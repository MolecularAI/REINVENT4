"""A simple bucket counter

Increments the count for a bucket when an item is added.  Can test if the bucket
is full.
"""

from __future__ import annotations

all = ["BucketCounter"]
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable


class BucketCounter(Counter):
    """Simple bucket counter

    Counts all occurrences of items by putting them in a bucket.  Each bucket
    can be checked if full.  Buckets themselves are not limited in size.
    """

    def __init__(self, max_size: int = 10, *args, **kwargs):
        """Bucket setup

        :param max_size: maximum size of bucket (for size check only)
        """
        super().__init__(*args, **kwargs)

        self.max_size = max_size

    def add(self, item: Hashable) -> None:
        """Add one item to its bucket

        NOTE: each bucket is of unlimited size

        :param item: the item to be added
        """

        self[item] += 1

    def bucket_full(self, item: Hashable) -> bool:
        """Check if bucket is full

        :param item: selects the bucket
        :return: whether bucket is full
        """
        if item not in self or self[item] <= self.max_size:
            return False

        return True

    def full_buckets(self) -> filter:
        """Return all buckets larger than the set maximum size"""
        return filter(lambda t: t[1] > self.max_size, self.items())

    def count_full(self) -> int:
        """Count how many buckets are above th maximum size"""
        return len(list(self.full_buckets()))

    def __reduce__(self):
        return self.__class__, (self.max_size, dict(self))


if __name__ == "__main__":
    bc = BucketCounter(5, A=6, B=7, C=5, D=4)
    print(bc)
    print(
        bc.bucket_full("A"),
        bc.bucket_full("B"),
        bc.bucket_full("C"),
        bc.bucket_full("D"),
    )
