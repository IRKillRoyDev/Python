"""
Chinese Remainder Theorem:
GCD ( Greatest Common Divisor ) or HCF ( Highest Common Factor )

If gcd(a, b) divides ra - rb, there exists an integer n such that
n = ra (mod a) and n = rb (mod b). If n1 and n2 are two such integers,
then n1 = n2 (mod lcm(a, b)). The moduli do not have to be coprime.

Algorithm :

1. Use the extended Euclid algorithm to find x, y and g such that
   a*x + b*y = g, where g = gcd(a, b).
2. If g does not divide rb - ra, there is no solution.
3. Otherwise n = ra + a * (((rb - ra) // g) * x), reduced modulo lcm(a, b).
"""

from __future__ import annotations


# Extended Euclid
def extended_euclid(a: int, b: int) -> tuple[int, int]:
    """
    >>> extended_euclid(10, 6)
    (-1, 2)

    >>> extended_euclid(7, 5)
    (-2, 3)

    """
    if b == 0:
        return (1, 0)
    (x, y) = extended_euclid(b, a % b)
    k = a // b
    return (y, x - k * y)


# Uses ExtendedEuclid to find inverses
def chinese_remainder_theorem(n1: int, r1: int, n2: int, r2: int) -> int:
    """
    >>> chinese_remainder_theorem(5,1,7,3)
    31

    Explanation : 31 is the smallest number such that
                (i)  When we divide it by 5, we get remainder 1
                (ii) When we divide it by 7, we get remainder 3

    >>> chinese_remainder_theorem(6,1,4,3)
    7
    >>> chinese_remainder_theorem(4, 2, 6, 4)
    10
    >>> chinese_remainder_theorem(6, 1, 4, 2)
    Traceback (most recent call last):
        ...
    ValueError: no solution: the remainders are inconsistent for these moduli

    """
    return _two_congruence_solution(n1, r1, n2, r2)


# Modular inverse exists only when the arguments are coprime. The second
# solver below therefore uses the same general congruence solver.


# This function find the inverses of a i.e., a^(-1)
def invert_modulo(a: int, n: int) -> int:
    """
    >>> invert_modulo(2, 5)
    3

    >>> invert_modulo(8,7)
    1
    >>> invert_modulo(6, 4)
    Traceback (most recent call last):
        ...
    ValueError: inverse does not exist because the arguments are not coprime

    """
    b, y = extended_euclid(a, n)
    gcd = a * b + n * y
    if gcd < 0:
        gcd = -gcd
        b = -b
    if gcd != 1:
        msg = "inverse does not exist because the arguments are not coprime"
        raise ValueError(msg)
    if b < 0:
        b = (b % n + n) % n
    return b


def chinese_remainder_theorem2(n1: int, r1: int, n2: int, r2: int) -> int:
    """
    >>> chinese_remainder_theorem2(5,1,7,3)
    31

    >>> chinese_remainder_theorem2(6,1,4,3)
    7

    """
    return _two_congruence_solution(n1, r1, n2, r2)


def _two_congruence_solution(n1: int, r1: int, n2: int, r2: int) -> int:
    """Return the smallest non-negative solution of the two congruences.

    ``extended_euclid`` returns coefficients for ``n1 * x + n2 * y = gcd``.
    The old formula treated that gcd as 1, so non-coprime moduli produced a
    number that did not satisfy either congruence. ``(6, 1, 4, 3)`` returned
    14, but ``14 % 6 == 2`` and ``14 % 4 == 2``. The solution is 7.
    """
    if n1 <= 0 or n2 <= 0:
        msg = "moduli must be positive integers"
        raise ValueError(msg)
    x, y = extended_euclid(n1, n2)
    gcd = n1 * x + n2 * y
    if gcd < 0:
        gcd = -gcd
        x = -x
    if (r2 - r1) % gcd != 0:
        msg = "no solution: the remainders are inconsistent for these moduli"
        raise ValueError(msg)
    modulus = n1 // gcd * n2
    value = r1 + n1 * (((r2 - r1) // gcd) * x)
    return value % modulus


if __name__ == "__main__":
    from doctest import testmod

    testmod(name="chinese_remainder_theorem", verbose=True)
    testmod(name="chinese_remainder_theorem2", verbose=True)
    testmod(name="invert_modulo", verbose=True)
    testmod(name="extended_euclid", verbose=True)
