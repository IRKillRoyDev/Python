"""
HyperLogLog.

HyperLogLog estimates how many distinct items were in a stream. It keeps one
small integer per register, the longest run of leading zeros seen in a hash
of an item that landed in that register. The harmonic mean of those values
is turned into a cardinality estimate.

https://en.wikipedia.org/wiki/HyperLogLog

Hashes are the first 4 bytes of MD5, so the estimate does not depend on
Python's hash salt. The result is approximate. Small cardinalities use the
empty-register correction from the original algorithm.
"""

from __future__ import annotations

import hashlib
import math


class HyperLogLog:
    """
    Estimate the number of distinct strings added so far.

    >>> sketch = HyperLogLog(precision=8)
    >>> sketch.estimate()
    0.0
    >>> for word in ("red", "green", "blue", "red"):
    ...     sketch.add(word)
    >>> 2 < sketch.estimate() < 4
    True
    >>> busy = HyperLogLog(precision=10)
    >>> for number in range(200):
    ...     busy.add(f"user-{number}")
    >>> 150 < busy.estimate() < 260
    True
    """

    def __init__(self, precision: int = 8) -> None:
        if not 4 <= precision <= 16:
            msg = "precision must be from 4 to 16"
            raise ValueError(msg)
        self.precision = precision
        self._registers = [0 for _ in range(1 << precision)]
        buckets = len(self._registers)
        if buckets == 16:
            self._alpha = 0.673
        elif buckets == 32:
            self._alpha = 0.697
        elif buckets == 64:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1 + 1.079 / buckets)

    def add(self, item: str) -> None:
        """Observe one occurrence of ``item``."""
        digest = hashlib.md5(item.encode(), usedforsecurity=False).digest()
        hashed = int.from_bytes(digest[:4], "big")
        index = hashed >> (32 - self.precision)
        width = 32 - self.precision
        remainder = hashed & ((1 << width) - 1)
        rank = width - remainder.bit_length() + 1
        self._registers[index] = max(self._registers[index], rank)

    def estimate(self) -> float:
        """Return the estimated number of distinct items."""
        buckets = len(self._registers)
        indicator = sum(2.0**-rank for rank in self._registers)
        raw = self._alpha * buckets * buckets / indicator
        if raw <= 2.5 * buckets:
            zeros = self._registers.count(0)
            if zeros:
                return buckets * math.log(buckets / zeros)
        limit = (1 << 32) / 30
        if raw > limit:
            return -(1 << 32) * math.log(1 - raw / (1 << 32))
        return raw


if __name__ == "__main__":
    import doctest

    doctest.testmod()
