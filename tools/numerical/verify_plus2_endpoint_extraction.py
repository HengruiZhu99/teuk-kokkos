#!/usr/bin/env python3
"""Independent 100-digit audit of constrained plus2 endpoint extraction."""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp


mp.mp.dps = 100


def solve(nodes: list[int], power: int) -> list[mp.mpf]:
    matrix = mp.matrix([[mp.mpf(node) ** k for node in nodes]
                        for k in range(len(nodes))])
    rhs = mp.matrix([int(k == power) for k in range(len(nodes))])
    return list(mp.lu_solve(matrix, rhs))


def main() -> int:
    expected0 = [Fraction(29, 6), Fraction(-461, 24), Fraction(31),
                 Fraction(-307, 12), Fraction(65, 6), Fraction(-15, 8)]
    expected1 = [Fraction(-77, 12), Fraction(107, 6), Fraction(-39, 2),
                 Fraction(61, 6), Fraction(-25, 12)]
    weights0 = solve(list(range(1, 7)), 2)
    weights1 = solve(list(range(1, 6)), 1)
    for actual, exact in zip(weights0, expected0):
        expected = mp.mpf(exact.numerator) / exact.denominator
        assert abs(actual - expected) < mp.mpf("1e-90")
    for actual, exact in zip(weights1, expected1):
        expected = mp.mpf(exact.numerator) / exact.denominator
        assert abs(actual - expected) < mp.mpf("1e-90")

    for spin in (mp.mpf("0"), mp.mpf("0.73"), mp.mpf("-0.73"),
                 mp.mpf("0.999")):
        previous = None
        errors = []
        for count in (9, 17, 33, 65):
            h = mp.mpf("0.42") / (count - 1)
            amplitude = mp.mpc("0.7", "-0.21")
            cosine = mp.mpf("0.6")

            def q0(radius: mp.mpf) -> mp.mpc:
                dm = mp.mpc(mp.mpf("1.7") ** 2, -spin * radius * cosine)
                return amplitude * mp.exp(mp.mpc("0.9", "-0.23") * radius) / dm**4

            def q1(radius: mp.mpf) -> mp.mpc:
                return (mp.mpc("0.3", "-0.17") + amplitude) * mp.exp(
                    mp.mpc("-0.4", "0.31") * radius)

            f0 = lambda r: r**2 * q0(r)
            f1 = lambda r: r * q1(r)
            actual0 = sum(w * f0((j + 1) * h)
                          for j, w in enumerate(weights0)) / h**2
            actual1 = sum(w * f1((j + 1) * h)
                          for j, w in enumerate(weights1)) / h
            error = max(abs(actual0 - q0(mp.mpf("0"))),
                        abs(actual1 - q1(mp.mpf("0"))))
            errors.append(error)
            if previous is not None:
                assert previous / error > 15
            previous = error
        print(f"PASS spin={spin} errors=" + ",".join(mp.nstr(e, 12) for e in errors))
    print("PASS exact moments, rational weights, 100-digit convergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
