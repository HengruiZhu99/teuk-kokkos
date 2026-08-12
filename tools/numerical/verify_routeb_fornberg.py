#!/usr/bin/env python3
"""Arbitrary-precision host audit for direct Route-B D1..D4 formulas."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import mpmath as mp

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from generate_routeb_fornberg_weights import (  # noqa: E402
    DERIVATIVES,
    WINDOW,
    all_weights,
    canonical_table,
)


EXPECTED_SHA256 = "b6114660d48020bc2d18313f0efc561bd05ed5c5a4386d4b6e129e118d8b5c5a"


def independent_weights(derivative: int, evaluation_position: int):
    offsets = [mp.mpf(column - evaluation_position) for column in range(WINDOW)]
    matrix = mp.matrix([[offset**power for offset in offsets]
                        for power in range(WINDOW)])
    rhs = mp.matrix([mp.factorial(derivative) if power == derivative else 0
                     for power in range(WINDOW)])
    return list(mp.lu_solve(matrix, rhs))


def independent_table():
    return [[independent_weights(derivative, row) for row in range(WINDOW)]
            for derivative in DERIVATIVES]


def apply(values: list[mp.mpc], spacing: mp.mpf, index: int,
          derivative: int, table) -> mp.mpc:
    start = min(max(index - WINDOW // 2, 0), len(values) - WINDOW)
    row = index - start
    result = mp.mpc(0)
    for column, weight in enumerate(table[derivative - 1][row]):
        result += weight * values[start + column]
    return result / spacing**derivative


def maximum_error(points: int, derivative: int, table, profile: str) -> mp.mpf:
    left, right = mp.mpf("-0.23"), mp.mpf("0.71")
    spacing = (right - left) / (points - 1)
    coordinates = [left + i * spacing for i in range(points)]
    alpha, beta = mp.mpf("1.37"), mp.mpf("2.41")
    if profile == "exponential":
        values = [mp.exp(alpha * x) for x in coordinates]
        exact = [alpha**derivative * mp.exp(alpha * x) for x in coordinates]
    else:
        values = [mp.sin(beta * x) for x in coordinates]
        exact = [beta**derivative * mp.sin(beta * x + derivative * mp.pi / 2)
                 for x in coordinates]
    return max(abs(apply(values, spacing, i, derivative, table) - exact[i])
               for i in range(points))


def main() -> None:
    mp.mp.dps = 100
    exact_table = all_weights()
    checksum = hashlib.sha256(canonical_table(exact_table).encode()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise AssertionError(f"Route-B weight checksum changed: {checksum}")

    # Exact moment conditions for every derivative and every position in the
    # shifted nine-point window.
    for derivative, derivative_table in zip(DERIVATIVES, exact_table):
        for row, weights in enumerate(derivative_table):
            for power in range(WINDOW):
                moment = sum(weight * (column - row) ** power
                             for column, weight in enumerate(weights))
                expected = math.factorial(derivative) if power == derivative else 0
                if moment != expected:
                    raise AssertionError(
                        f"moment mismatch D{derivative} row {row} power {power}")

    # Reconstruct every row independently with a high-precision Vandermonde
    # solve rather than reusing the Fraction elimination implementation.
    table = independent_table()
    for derivative_index, derivative_table in enumerate(exact_table):
        for row, weights in enumerate(derivative_table):
            for column, weight in enumerate(weights):
                expected = mp.mpf(weight.numerator) / weight.denominator
                difference = abs(
                    table[derivative_index][row][column] - expected)
                if difference > mp.mpf("1e-90"):
                    raise AssertionError(
                        "independent weight mismatch "
                        f"D{derivative_index + 1} row {row} column {column}")

    for profile in ("exponential", "trigonometric"):
        for derivative in DERIVATIVES:
            errors = [maximum_error(points, derivative, table, profile)
                      for points in (17, 33, 65)]
            ratios = (errors[0] / errors[1], errors[1] / errors[2])
            if min(ratios) <= mp.mpf("15"):
                raise AssertionError(
                    f"sub-fourth-order {profile} D{derivative}: {ratios}")
            print(
                f"Route-B {profile} D{derivative} max errors "
                f"{mp.nstr(errors[0], 8)} {mp.nstr(errors[1], 8)} "
                f"{mp.nstr(errors[2], 8)} ratios "
                f"{mp.nstr(ratios[0], 7)} {mp.nstr(ratios[1], 7)}"
            )

    for derivative, derivative_table in zip(DERIVATIVES, table):
        l1 = max(sum(abs(value) for value in row) for row in derivative_table)
        l2 = max(mp.sqrt(sum(value**2 for value in row))
                 for row in derivative_table)
        print(f"Route-B D{derivative} dimensionless max L1/L2 norms "
              f"{mp.nstr(l1, 12)} {mp.nstr(l2, 12)}")

    print("Route-B direct Fornberg arbitrary-precision audit: PASS")


if __name__ == "__main__":
    main()
