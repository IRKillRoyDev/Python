"""
AES-128 for a single 16-byte block.

The Advanced Encryption Standard is a symmetric block cipher. This file
implements the 128-bit-key version: SubBytes, ShiftRows, MixColumns,
AddRoundKey, and the key schedule from FIPS 197.

https://en.wikipedia.org/wiki/Advanced_Encryption_Standard

This is a teaching implementation of one block. It has no cipher mode, no
padding, and no authentication. Do not use it to protect real data.
"""

from __future__ import annotations

_SBOX = list(
    bytes.fromhex(
        "637c777bf26b6fc53001672bfed7ab76"
        "ca82c97dfa5947f0add4a2af9ca472c0"
        "b7fd9326363ff7cc34a5e5f171d83115"
        "04c723c31896059a071280e2eb27b275"
        "09832c1a1b6e5aa0523bd6b329e32f84"
        "53d100ed20fcb15b6acbbe394a4c58cf"
        "d0efaafb434d338545f9027f503c9fa8"
        "51a3408f929d38f5bcb6da2110fff3d2"
        "cd0c13ec5f974417c4a77e3d645d1973"
        "60814fdc222a908846eeb814de5e0bdb"
        "e0323a0a4906245cc2d3ac629195e479"
        "e7c8376d8dd54ea96c56f4ea657aae08"
        "ba78252e1ca6b4c6e8dd741f4bbd8b8a"
        "703eb5664803f60e613557b986c11d9e"
        "e1f8981169d98e949b1e87e9ce5528df"
        "8ca1890dbfe6426841992d0fb054bb16"
    )
)
_INV_SBOX = [0] * 256
for _index, _value in enumerate(_SBOX):
    _INV_SBOX[_value] = _index

_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)
_BLOCK_BYTES = 16
_ROUNDS = 10


def aes128_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt one 16-byte block with a 16-byte key.

    The expected ciphertext is the FIPS 197 appendix B vector, and the second
    call is the all-zero NIST vector.

    >>> key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    >>> plain = bytes.fromhex("00112233445566778899aabbccddeeff")
    >>> aes128_encrypt(plain, key).hex()
    '69c4e0d86a7b0430d8cdb78070b4c55a'
    >>> aes128_encrypt(bytes(16), bytes(16)).hex()
    '66e94bd4ef8a2c3b884cfa59ca342b2e'
    >>> aes128_encrypt(b"short", key)
    Traceback (most recent call last):
        ...
    ValueError: AES-128 works on a single 16-byte block
    """
    state = _as_block(plaintext)
    round_keys = _expand_key(_as_block(key))
    _add_round_key(state, round_keys[0])
    for round_index in range(1, _ROUNDS):
        _sub_bytes(state, _SBOX)
        _shift_rows(state)
        _mix_columns(state, inverse=False)
        _add_round_key(state, round_keys[round_index])
    _sub_bytes(state, _SBOX)
    _shift_rows(state)
    _add_round_key(state, round_keys[_ROUNDS])
    return bytes(state)


def aes128_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """
    Invert ``aes128_encrypt`` for one block.

    >>> key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    >>> plain = bytes.fromhex("00112233445566778899aabbccddeeff")
    >>> cipher = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    >>> aes128_decrypt(cipher, key) == plain
    True
    >>> aes128_decrypt(aes128_encrypt(bytes(16), bytes(16)), bytes(16)) == bytes(16)
    True
    """
    state = _as_block(ciphertext)
    round_keys = _expand_key(_as_block(key))
    _add_round_key(state, round_keys[_ROUNDS])
    for round_index in range(_ROUNDS - 1, 0, -1):
        _inv_shift_rows(state)
        _sub_bytes(state, _INV_SBOX)
        _add_round_key(state, round_keys[round_index])
        _mix_columns(state, inverse=True)
    _inv_shift_rows(state)
    _sub_bytes(state, _INV_SBOX)
    _add_round_key(state, round_keys[0])
    return bytes(state)


def _as_block(block: bytes) -> list[int]:
    if len(block) != _BLOCK_BYTES:
        msg = "AES-128 works on a single 16-byte block"
        raise ValueError(msg)
    return list(block)


def _xtime(value: int) -> int:
    shifted = (value << 1) & 0xFF
    if value & 0x80:
        shifted ^= 0x1B
    return shifted


def _mul(left: int, right: int) -> int:
    product = 0
    for _ in range(8):
        if right & 1:
            product ^= left
        left = _xtime(left)
        right >>= 1
    return product


def _sub_bytes(state: list[int], box: list[int]) -> None:
    for index, value in enumerate(state):
        state[index] = box[value]


def _shift_rows(state: list[int]) -> None:
    for row in range(1, 4):
        cells = [state[row + 4 * column] for column in range(4)]
        rotated = cells[row:] + cells[:row]
        for column, value in enumerate(rotated):
            state[row + 4 * column] = value


def _inv_shift_rows(state: list[int]) -> None:
    for row in range(1, 4):
        cells = [state[row + 4 * column] for column in range(4)]
        rotated = cells[-row:] + cells[:-row]
        for column, value in enumerate(rotated):
            state[row + 4 * column] = value


def _mix_column(column: list[int]) -> list[int]:
    a0, a1, a2, a3 = column
    return [
        _mul(a0, 2) ^ _mul(a1, 3) ^ a2 ^ a3,
        a0 ^ _mul(a1, 2) ^ _mul(a2, 3) ^ a3,
        a0 ^ a1 ^ _mul(a2, 2) ^ _mul(a3, 3),
        _mul(a0, 3) ^ a1 ^ a2 ^ _mul(a3, 2),
    ]


def _inv_mix_column(column: list[int]) -> list[int]:
    a0, a1, a2, a3 = column
    return [
        _mul(a0, 14) ^ _mul(a1, 11) ^ _mul(a2, 13) ^ _mul(a3, 9),
        _mul(a0, 9) ^ _mul(a1, 14) ^ _mul(a2, 11) ^ _mul(a3, 13),
        _mul(a0, 13) ^ _mul(a1, 9) ^ _mul(a2, 14) ^ _mul(a3, 11),
        _mul(a0, 11) ^ _mul(a1, 13) ^ _mul(a2, 9) ^ _mul(a3, 14),
    ]


def _mix_columns(state: list[int], *, inverse: bool) -> None:
    mix = _inv_mix_column if inverse else _mix_column
    for column in range(4):
        start = 4 * column
        state[start : start + 4] = mix(state[start : start + 4])


def _add_round_key(state: list[int], round_key: list[int]) -> None:
    for index, value in enumerate(round_key):
        state[index] ^= value


def _expand_key(key: list[int]) -> list[list[int]]:
    words = [key[index : index + 4] for index in range(0, _BLOCK_BYTES, 4)]
    for index in range(4, 4 * (_ROUNDS + 1)):
        temp = words[-1][:]
        if index % 4 == 0:
            temp = temp[1:] + temp[:1]
            temp = [_SBOX[byte] for byte in temp]
            temp[0] ^= _RCON[index // 4 - 1]
        earlier = words[index - 4]
        words.append([earlier[byte] ^ temp[byte] for byte in range(4)])
    round_keys: list[list[int]] = []
    for round_index in range(_ROUNDS + 1):
        block: list[int] = []
        for word in words[4 * round_index : 4 * round_index + 4]:
            block.extend(word)
        round_keys.append(block)
    return round_keys


if __name__ == "__main__":
    import doctest

    doctest.testmod()
