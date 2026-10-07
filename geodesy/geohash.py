"""
Geohash.

Geohash interleaves longitude and latitude bits and writes them as a base32
string. A longer string names a smaller cell. Nearby points often share a
prefix, which is why the encoding is used as a spatial index key.

https://en.wikipedia.org/wiki/Geohash

The alphabet omits a, i, l, and o. ``encode`` followed by ``decode`` returns
the center of the cell that contains the original point.
"""

from __future__ import annotations

_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {character: index for index, character in enumerate(_ALPHABET)}


def encode(latitude: float, longitude: float, precision: int = 12) -> str:
    """
    Encode a WGS84 point as a geohash of ``precision`` characters.

    >>> encode(57.64911, 10.40744, precision=11)
    'u4pruydqqvj'
    >>> encode(0, 0, precision=1)
    's'
    >>> encode(91, 0)
    Traceback (most recent call last):
        ...
    ValueError: latitude must be from -90 to 90
    """
    if precision < 1:
        msg = "precision must be at least 1"
        raise ValueError(msg)
    if not -90 <= latitude <= 90:
        msg = "latitude must be from -90 to 90"
        raise ValueError(msg)
    if not -180 <= longitude <= 180:
        msg = "longitude must be from -180 to 180"
        raise ValueError(msg)
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    is_longitude = True
    while len(bits) < precision * 5:
        if is_longitude:
            _bisect(longitude, lon_range, bits)
        else:
            _bisect(latitude, lat_range, bits)
        is_longitude = not is_longitude
    characters: list[str] = []
    for start in range(0, len(bits), 5):
        index = 0
        for bit in bits[start : start + 5]:
            index = (index << 1) | bit
        characters.append(_ALPHABET[index])
    return "".join(characters)


def decode(geohash: str) -> tuple[float, float]:
    """
    Return the center ``(latitude, longitude)`` of a geohash cell.

    >>> latitude, longitude = decode("u4pruydqqvj")
    >>> round(latitude, 5), round(longitude, 5)
    (57.64911, 10.40744)
    >>> decode("s")
    (22.5, 22.5)
    >>> decode("")
    Traceback (most recent call last):
        ...
    ValueError: geohash must not be empty
    """
    if geohash == "":
        msg = "geohash must not be empty"
        raise ValueError(msg)
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_longitude = True
    for character in geohash:
        try:
            value = _DECODE[character]
        except KeyError as exc:
            msg = "geohash contains a character outside the base32 alphabet"
            raise ValueError(msg) from exc
        for shift in range(4, -1, -1):
            bit = (value >> shift) & 1
            interval = lon_range if is_longitude else lat_range
            midpoint = (interval[0] + interval[1]) / 2
            if bit:
                interval[0] = midpoint
            else:
                interval[1] = midpoint
            is_longitude = not is_longitude
    latitude = (lat_range[0] + lat_range[1]) / 2
    longitude = (lon_range[0] + lon_range[1]) / 2
    return latitude, longitude


def _bisect(value: float, interval: list[float], bits: list[int]) -> None:
    midpoint = (interval[0] + interval[1]) / 2
    if value >= midpoint:
        bits.append(1)
        interval[0] = midpoint
    else:
        bits.append(0)
        interval[1] = midpoint


if __name__ == "__main__":
    import doctest

    doctest.testmod()
