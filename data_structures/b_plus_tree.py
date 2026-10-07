"""
B+ tree.

A B+ tree stores every key in a leaf. Internal nodes only hold separators
that route a search, and the leaves are linked so a range can be scanned
without walking back up the tree. That is the layout of most database indexes.

https://en.wikipedia.org/wiki/B%2B_tree

``order`` is the maximum number of keys in a node. Inserting a duplicate
does nothing. Deletion is not implemented: this tree only grows.
"""

from __future__ import annotations


class _Node:
    def __init__(self, leaf: bool) -> None:
        self.leaf = leaf
        self.keys: list[int] = []
        self.children: list[_Node] = []
        self.next: _Node | None = None


class BPlusTree:
    """
    Insert, point lookup, and inclusive range scan.

    >>> tree = BPlusTree(order=3)
    >>> for key in (10, 20, 5, 6, 12, 30, 7, 17):
    ...     tree.insert(key)
    >>> tree.keys()
    [5, 6, 7, 10, 12, 17, 20, 30]
    >>> tree.contains(17)
    True
    >>> tree.contains(18)
    False
    >>> tree.range_query(6, 17)
    [6, 7, 10, 12, 17]
    >>> tree.insert(10)
    >>> tree.keys()
    [5, 6, 7, 10, 12, 17, 20, 30]
    """

    def __init__(self, order: int = 4) -> None:
        if order < 3:
            msg = "order must be at least 3"
            raise ValueError(msg)
        self.order = order
        self.root = _Node(leaf=True)

    def contains(self, key: int) -> bool:
        """Return whether ``key`` is stored in a leaf."""
        return key in _find_leaf(self.root, key).keys

    def insert(self, key: int) -> None:
        """Insert ``key`` into the correct leaf, splitting nodes that overflow."""
        leaf = _find_leaf(self.root, key)
        if not _insert_leaf(leaf, key):
            return
        if len(leaf.keys) > self.order:
            self._split_leaf(leaf)

    def keys(self) -> list[int]:
        """Return every key in sorted order by walking the leaf chain."""
        node = self.root
        while not node.leaf:
            node = node.children[0]
        found: list[int] = []
        while node is not None:
            found.extend(node.keys)
            node = node.next
        return found

    def range_query(self, start: int, end: int) -> list[int]:
        """Return keys ``k`` with ``start <= k <= end``."""
        if start > end:
            return []
        node: _Node | None = _find_leaf(self.root, start)
        found: list[int] = []
        while node is not None:
            for key in node.keys:
                if key > end:
                    return found
                if key >= start:
                    found.append(key)
            node = node.next
        return found

    def _split_leaf(self, leaf: _Node) -> None:
        midpoint = len(leaf.keys) // 2
        right = _Node(leaf=True)
        right.keys = leaf.keys[midpoint:]
        leaf.keys = leaf.keys[:midpoint]
        right.next = leaf.next
        leaf.next = right
        self._insert_into_parent(leaf, right.keys[0], right)

    def _split_internal(self, node: _Node) -> None:
        midpoint = len(node.keys) // 2
        promoted = node.keys[midpoint]
        right = _Node(leaf=False)
        right.keys = node.keys[midpoint + 1 :]
        right.children = node.children[midpoint + 1 :]
        node.keys = node.keys[:midpoint]
        node.children = node.children[: midpoint + 1]
        self._insert_into_parent(node, promoted, right)

    def _insert_into_parent(self, left: _Node, key: int, right: _Node) -> None:
        if left is self.root:
            parent = _Node(leaf=False)
            parent.keys = [key]
            parent.children = [left, right]
            self.root = parent
            return
        parent = _find_parent(self.root, left)
        index = parent.children.index(left)
        parent.keys.insert(index, key)
        parent.children.insert(index + 1, right)
        if len(parent.keys) > self.order:
            self._split_internal(parent)


def _find_leaf(node: _Node, key: int) -> _Node:
    while not node.leaf:
        index = 0
        while index < len(node.keys) and key >= node.keys[index]:
            index += 1
        node = node.children[index]
    return node


def _insert_leaf(leaf: _Node, key: int) -> bool:
    if key in leaf.keys:
        return False
    index = 0
    while index < len(leaf.keys) and leaf.keys[index] < key:
        index += 1
    leaf.keys.insert(index, key)
    return True


def _find_parent(node: _Node, target: _Node) -> _Node:
    if node.leaf:
        msg = "leaf has no parent in this subtree"
        raise ValueError(msg)
    for child in node.children:
        if child is target:
            return node
    for child in node.children:
        if not child.leaf:
            try:
                return _find_parent(child, target)
            except ValueError:
                continue
    msg = "node is not in the tree"
    raise ValueError(msg)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
