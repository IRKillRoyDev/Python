"""
Consistent hashing.

Keys and nodes are placed on a ring. A key is owned by the first node
clockwise from its position. Adding or removing a node moves only the keys
that lived in that node's arc, which is why caches and shard maps use it.

https://en.wikipedia.org/wiki/Consistent_hashing

Positions come from MD5 so the assignment does not depend on Python's
per-process hash salt. ``replicas`` is the number of virtual nodes per
physical node.
"""

from __future__ import annotations

import bisect
import hashlib


class ConsistentHash:
    """
    Map keys onto a set of nodes.

    >>> ring = ConsistentHash(["cache-a", "cache-b", "cache-c"], replicas=8)
    >>> ring.get_node("user-42")
    'cache-b'
    >>> ring.get_node("user-42")
    'cache-b'
    >>> ring.remove_node("cache-b")
    >>> ring.get_node("user-42")
    'cache-c'
    >>> ring.add_node("cache-b")
    >>> sorted(ring.nodes())
    ['cache-a', 'cache-b', 'cache-c']
    """

    def __init__(self, nodes: list[str] | None = None, replicas: int = 100) -> None:
        if replicas < 1:
            msg = "replicas must be at least 1"
            raise ValueError(msg)
        self.replicas = replicas
        self._ring: dict[int, str] = {}
        self._keys: list[int] = []
        for node in nodes or []:
            self.add_node(node)

    def add_node(self, node: str) -> None:
        """Place ``replicas`` positions for ``node`` on the ring."""
        for replica in range(self.replicas):
            position = _hash(f"{node}:{replica}")
            if position in self._ring:
                continue
            self._ring[position] = node
            bisect.insort(self._keys, position)

    def remove_node(self, node: str) -> None:
        """Remove every virtual position that belongs to ``node``."""
        for replica in range(self.replicas):
            position = _hash(f"{node}:{replica}")
            owner = self._ring.get(position)
            if owner != node:
                continue
            del self._ring[position]
            index = bisect.bisect_left(self._keys, position)
            del self._keys[index]

    def get_node(self, key: str) -> str:
        """Return the node that owns ``key``."""
        if not self._keys:
            msg = "ring has no nodes"
            raise ValueError(msg)
        position = _hash(key)
        index = bisect.bisect_left(self._keys, position) % len(self._keys)
        return self._ring[self._keys[index]]

    def nodes(self) -> set[str]:
        """Return the physical nodes currently on the ring."""
        return set(self._ring.values())


def _hash(value: str) -> int:
    digest = hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
