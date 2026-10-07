"""
Reservoir sampling.

Algorithm R keeps a uniform random sample of ``k`` items from a stream while
looking at each item once. The stream length does not need to be known ahead
of time, which is the case for logs and other data that does not fit in memory.

https://en.wikipedia.org/wiki/Reservoir_sampling

Pass a ``random.Random`` instance when a caller needs a repeatable sample.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator


def reservoir_sample[T](
    stream: Iterable[T], k: int, rng: random.Random | None = None
) -> list[T]:
    """
    Return ``k`` items drawn uniformly from ``stream``.

    If the stream is shorter than ``k``, every item is returned.

    >>> sample = reservoir_sample(range(10), 4, random.Random(0))
    >>> sample
    [6, 1, 2, 5]
    >>> sorted(sample) == sorted(set(sample))
    True
    >>> reservoir_sample(["a", "b"], 5, random.Random(1))
    ['a', 'b']
    >>> reservoir_sample([1, 2, 3], 0, random.Random(1))
    []
    """
    if k < 0:
        msg = "sample size k must be non-negative"
        raise ValueError(msg)
    generator = rng if rng is not None else random.Random()
    reservoir: list[T] = []
    for index, item in enumerate(stream):
        if index < k:
            reservoir.append(item)
            continue
        chosen = generator.randrange(index + 1)
        if chosen < k:
            reservoir[chosen] = item
    return reservoir


def reservoir_stream[T](stream: Iterator[T], k: int) -> list[T]:
    """
    Sample ``stream`` with a fresh generator. See ``reservoir_sample``.

    >>> len(reservoir_stream(iter(["a", "b", "c"]), 2))
    2
    """
    return reservoir_sample(stream, k)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
