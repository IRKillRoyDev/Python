"""
Count-Min sketch.

A small table of counters estimates how many times an item appeared in a
stream. The estimate is never smaller than the true count. Collisions can
make it larger. That trade is what lets a metrics system count events it
cannot store one by one.

https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch
"""

from __future__ import annotations

import hashlib


class CountMinSketch:
    """
    Add events and read a count estimate.

    >>> sketch = CountMinSketch(width=64, depth=4)
    >>> for _ in range(5):
    ...     sketch.add("login")
    >>> sketch.add("login", 2)
    >>> sketch.estimate("login")
    7
    >>> sketch.estimate("missing")
    0
    >>> tiny = CountMinSketch(width=2, depth=1)
    >>> tiny.add("a")
    >>> tiny.add("b")
    >>> tiny.estimate("a") >= 1
    True
    """

    def __init__(self, width: int = 128, depth: int = 4) -> None:
        if width < 1 or depth < 1:
            msg = "width and depth must be at least 1"
            raise ValueError(msg)
        self.width = width
        self.depth = depth
        self._table = [[0 for _ in range(width)] for _ in range(depth)]

    def add(self, item: str, count: int = 1) -> None:
        """Increase the estimated count of ``item`` by ``count``."""
        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)
        for row, column in enumerate(self._indexes(item)):
            self._table[row][column] += count

    def estimate(self, item: str) -> int:
        """Return an estimate that is at least the true added count."""
        return min(
            self._table[row][column] for row, column in enumerate(self._indexes(item))
        )

    def _indexes(self, item: str) -> list[int]:
        indexes: list[int] = []
        for row in range(self.depth):
            payload = f"{row}:{item}".encode()
            digest = hashlib.md5(payload, usedforsecurity=False).digest()
            indexes.append(int.from_bytes(digest[:4], "big") % self.width)
        return indexes


if __name__ == "__main__":
    import doctest

    doctest.testmod()
