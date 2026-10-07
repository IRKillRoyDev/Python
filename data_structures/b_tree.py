"""
B-tree.

A B-tree keeps keys sorted in a balanced tree where every node can hold more
than one key. Databases use that shape so a node lines up with a disk page.

https://en.wikipedia.org/wiki/B-tree

This tree is a set: inserting a key that is already present does nothing.
``t`` is the minimum degree from CLRS. A node holds at most ``2 * t - 1`` keys.
"""

from __future__ import annotations

from collections.abc import Iterator


class _Node:
    def __init__(self, leaf: bool) -> None:
        self.keys: list[int] = []
        self.children: list[_Node] = []
        self.leaf = leaf


class BTree:
    """
    Search and insert on a B-tree.

    >>> tree = BTree(t=2)
    >>> for key in (10, 20, 5, 6, 12, 30, 7, 17):
    ...     tree.insert(key)
    >>> tree.keys()
    [5, 6, 7, 10, 12, 17, 20, 30]
    >>> tree.contains(12)
    True
    >>> tree.contains(13)
    False
    >>> tree.insert(12)
    >>> tree.keys()
    [5, 6, 7, 10, 12, 17, 20, 30]
    """

    def __init__(self, t: int = 2) -> None:
        if t < 2:
            msg = "minimum degree t must be at least 2"
            raise ValueError(msg)
        self.t = t
        self.root = _Node(leaf=True)

    def contains(self, key: int) -> bool:
        """Return whether ``key`` is stored in the tree."""
        return _search(self.root, key)

    def insert(self, key: int) -> None:
        """Insert ``key``, splitting full nodes on the way down."""
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            grown = _Node(leaf=False)
            grown.children.append(root)
            _split_child(grown, 0, self.t)
            self.root = grown
        _insert_non_full(self.root, key, self.t)

    def keys(self) -> list[int]:
        """Return every key in sorted order."""
        return list(_walk(self.root))


def _search(node: _Node, key: int) -> bool:
    index = 0
    while index < len(node.keys) and key > node.keys[index]:
        index += 1
    if index < len(node.keys) and node.keys[index] == key:
        return True
    if node.leaf:
        return False
    return _search(node.children[index], key)


def _split_child(parent: _Node, index: int, degree: int) -> None:
    full = parent.children[index]
    sibling = _Node(leaf=full.leaf)
    sibling.keys = full.keys[degree:]
    median = full.keys[degree - 1]
    full.keys = full.keys[: degree - 1]
    if not full.leaf:
        sibling.children = full.children[degree:]
        full.children = full.children[:degree]
    parent.children.insert(index + 1, sibling)
    parent.keys.insert(index, median)


def _insert_non_full(node: _Node, key: int, degree: int) -> None:
    if key in node.keys:
        return
    if node.leaf:
        index = 0
        while index < len(node.keys) and node.keys[index] < key:
            index += 1
        node.keys.insert(index, key)
        return
    index = 0
    while index < len(node.keys) and key > node.keys[index]:
        index += 1
    child = node.children[index]
    if len(child.keys) == 2 * degree - 1:
        _split_child(node, index, degree)
        if key == node.keys[index]:
            return
        if key > node.keys[index]:
            index += 1
    _insert_non_full(node.children[index], key, degree)


def _walk(node: _Node) -> Iterator[int]:
    if node.leaf:
        yield from node.keys
        return
    for index, key in enumerate(node.keys):
        yield from _walk(node.children[index])
        yield key
    yield from _walk(node.children[-1])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
