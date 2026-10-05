"""
Suffix array and longest common prefix array.

A suffix array is the starting indexes of every suffix of a text, sorted in
lexicographic order. The LCP array stores how many leading characters two
neighboring suffixes share. Together they answer "where does this pattern
occur" with a binary search instead of a scan.

https://en.wikipedia.org/wiki/Suffix_array

The array is built by sorting suffixes directly, which is the clearest correct
construction. The LCP array uses Kasai's algorithm.
"""

from __future__ import annotations


def build_suffix_array(text: str) -> list[int]:
    """
    Return the suffix array of ``text``.

    >>> build_suffix_array("banana")
    [5, 3, 1, 0, 4, 2]
    >>> build_suffix_array("")
    []
    >>> build_suffix_array("a")
    [0]
    """
    return sorted(range(len(text)), key=lambda start: text[start:])


def build_lcp_array(text: str, suffix_array: list[int]) -> list[int]:
    """
    Return the LCP array for ``text`` and its suffix array.

    Entry ``i`` is the shared prefix length of the suffixes at
    ``suffix_array[i - 1]`` and ``suffix_array[i]``. Entry 0 is 0.

    >>> text = "banana"
    >>> suffix_array = build_suffix_array(text)
    >>> build_lcp_array(text, suffix_array)
    [0, 1, 3, 0, 0, 2]
    >>> build_lcp_array("", [])
    []
    """
    length = len(text)
    if len(suffix_array) != length:
        msg = "suffix array length must match the text"
        raise ValueError(msg)
    if length == 0:
        return []
    rank = [0 for _ in range(length)]
    for position, start in enumerate(suffix_array):
        rank[start] = position
    longest = 0
    lcp = [0 for _ in range(length)]
    for start in range(length):
        position = rank[start]
        if position == 0:
            longest = 0
            continue
        previous = suffix_array[position - 1]
        while (
            start + longest < length
            and previous + longest < length
            and text[start + longest] == text[previous + longest]
        ):
            longest += 1
        lcp[position] = longest
        if longest:
            longest -= 1
    return lcp


def find_pattern(text: str, pattern: str, suffix_array: list[int]) -> list[int]:
    """
    Return the starting indexes where ``pattern`` occurs in ``text``.

    >>> text = "banana"
    >>> suffix_array = build_suffix_array(text)
    >>> find_pattern(text, "ana", suffix_array)
    [1, 3]
    >>> find_pattern(text, "xyz", suffix_array)
    []
    >>> find_pattern(text, "", suffix_array)
    []
    """
    if pattern == "":
        return []
    left = _lower_bound(text, pattern, suffix_array)
    right = _lower_bound(text, pattern + "\uffff", suffix_array)
    return sorted(suffix_array[index] for index in range(left, right))


def _lower_bound(text: str, pattern: str, suffix_array: list[int]) -> int:
    low = 0
    high = len(suffix_array)
    while low < high:
        mid = (low + high) // 2
        if text[suffix_array[mid] :] < pattern:
            low = mid + 1
        else:
            high = mid
    return low


if __name__ == "__main__":
    import doctest

    doctest.testmod()
