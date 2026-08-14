#!/usr/bin/env python3
"""Independent Q30 decay oracle + deterministic binary fixtures for Metal bridge audit."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
ROUND = 1 << 29
MASK32 = (1 << 32) - 1


def i32(word: int) -> int:
    word &= MASK32
    return word - (1 << 32) if word & (1 << 31) else word


def q30_decay(cell: int, coeff: int) -> tuple[int, int]:
    """Spec directly: unbounded signed product, +2^29, floor-divide 2^30, clamp."""
    q = (i32(cell) * i32(coeff) + ROUND) // (1 << 30)
    if q < I32_MIN:
        return I32_MIN, 1
    if q > I32_MAX:
        return I32_MAX, 1
    return q, 0


def lcg(seed: int):
    while True:
        seed = (1664525 * seed + 1013904223) & MASK32
        yield i32(seed)


def cases() -> list[tuple[str, int, list[int]]]:
    extrema = [0, 1, -1, 2, -2, (1 << 29), -(1 << 29), (1 << 30), -(1 << 30), I32_MAX, I32_MIN]
    out = [(f"count-{n}", c, extrema[:n]) for c in (0, 1, -1, (1 << 30), -(1 << 30), I32_MAX, I32_MIN) for n in range(10)]
    # Negative/adversarial coefficients: signs, near-Q30, and full signed extrema.
    out += [("negative-adversarial", c, extrema) for c in (-1, -2, -(1 << 29), -(1 << 30), I32_MIN, I32_MAX)]
    rng = lcg(0x56781234)
    out.append(("random-100003", I32_MAX, [next(rng) for _ in range(100003)]))
    return out


def emit(path: Path) -> None:
    # Little endian wire: magic, records(name length/name, coeff, count, input i32[], expected i32[], saturation).
    with path.open("wb") as f:
        f.write(b"Q30METAL1\0")
        for name, coeff, cells in cases():
            encoded = name.encode("ascii")
            expected = [q30_decay(x, coeff)[0] for x in cells]
            saturated = sum(q30_decay(x, coeff)[1] for x in cells)
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<iI", coeff, len(cells)))
            f.write(struct.pack(f"<{len(cells)}i", *cells))
            f.write(struct.pack(f"<{len(cells)}iI", *expected, saturated))
        f.write(struct.pack("<H", 0))


def check() -> None:
    probes = [
        (I32_MAX, I32_MAX, I32_MAX, 1),
        (I32_MIN, I32_MAX, I32_MIN, 1),
        (I32_MIN, I32_MIN, I32_MAX, 1),
        (-1, 1 << 29, 0, 0),  # (-2^29 + 2^29) >> 30: tie rises to zero.
        (-1, -(1 << 29), 1, 0),
    ]
    for cell, coeff, want, sat in probes:
        got = q30_decay(cell, coeff)
        if got != (want, sat):
            raise AssertionError((cell, coeff, got, want, sat))
    if len(cases()[-1][2]) != 100003:
        raise AssertionError("random count")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emit", type=Path)
    ns = p.parse_args()
    check()
    if ns.emit:
        emit(ns.emit)
    print("q30 metal oracle: scalar formula, count 0..9, extrema, negative coeffs, random 100003=ok")
