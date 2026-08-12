#!/usr/bin/env python3
"""Independent exact audit of the evolved spin +2 source normalization.

The raw source ledger naturally exposes S0/R^6.  The common compact equation,
however, is written for Z_plus after the complete field factor

    Psi0 = R^5 Z_plus/(L^2-i a R cos(theta))^4

has been divided out.  This script derives the raw TT coefficient directly
from the explicit code tetrad, proves that the historical 2 Sigma/R multiplier
leaves an unwanted factor, and derives the cancellation-safe S0/R^7 source.
"""

from __future__ import annotations

import random
import csv
from pathlib import Path

import sympy as sp


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "PLUS2_SOURCE_NORMALIZATION_LEDGER.csv").open(newline="") as stream:
        ledger = list(csv.DictReader(stream))
    check("normalization ledger has five named rows",
          [row["term_id"] for row in ledger] ==
          ["rj01", "ak01", "ak02", "q01", "forcing"])
    check("normalization ledger versions evolved source",
          "normalization version 2" in ledger[-1]["notes"] and
          "S0/R7" in ledger[-1]["regular_S0_over_R7"])
    R, M, a, y, L2 = sp.symbols("R M a y L2", real=True, finite=True)
    CT = sp.symbols("C_T", real=True)
    I = sp.I
    D = L2**2 + a**2 * R**2 * y**2
    dm = L2 - I * a * R * y
    dp = L2 + I * a * R * y
    sin2 = 1 - y**2
    E = L2 - 2 * M * R + a**2 * R**2 / L2

    # Ripley et al. Eq. (5) code-tetrad T components.  The raw principal
    # coefficient of (D ...)(Delta ...) - (delta ...)(bardelta ...) is
    # l^T n^T-m^T mbar^T.
    lT = R**2 * 2 * M * (2 * M - a**2 * R / L2) / D
    nT = 2 + 4 * M * R / L2
    mT = -I * a * R * sp.sqrt(sin2) / (sp.sqrt(2) * dm)
    mbarT = I * a * R * sp.sqrt(sin2) / (sp.sqrt(2) * dp)
    raw_tt = sp.factor(lT * nT - mT * mbarT)
    CT_explicit = (
        8 * M * (2 * M - a**2 * R / L2) * (1 + 2 * M * R / L2)
        - a**2 * sin2
    )
    check("raw tetrad TT coefficient",
          sp.simplify(raw_tt - R**2 * CT_explicit / (2 * D)) == 0)

    # Complete field normalization.  W=R f, with f not equal to one.
    W = R**5 / dm**4
    f = R**4 / dm**4
    check("W_plus equals R f", sp.simplify(W - R * f) == 0)
    N = 2 * D / R**3
    Araw = R**2 * CT / (2 * D)
    historical = sp.factor(N * Araw * R * f)
    corrected = sp.factor((N / f) * Araw * R * f)
    check("historical normalization leaves f", sp.simplify(historical - CT * f) == 0)
    check("historical normalization is not common CT", sp.simplify(historical - CT) != 0)
    check("complete field normalization gives common CT", sp.simplify(corrected - CT) == 0)
    check("corrected raw multiplier",
          sp.simplify(N / f - 2 * D * dm**4 / R**7) == 0)

    # Exact optical cancellation in the raw S0/R^6 radial J family.
    b = -sp.Rational(1, 2) * E / D
    rho0 = -sp.Rational(1, 2) * (L2**2 - 2 * L2 * M * R + a**2 * R**2) / (dm**2 * dp)
    rhobar0 = sp.conjugate(rho0).xreplace(
        {sp.conjugate(R): R, sp.conjugate(M): M,
         sp.conjugate(a): a, sp.conjugate(y): y,
         sp.conjugate(L2): L2}
    )
    cancellation = I * a * y * E * (3 * L2 + 5 * I * a * R * y) / (2 * D**2)
    check("optical cancellation identity",
          sp.factor(sp.cancel((5 * b - 4 * rho0 - rhobar0) / R - cancellation)) == 0)
    check("optical cancellation finite at scri",
          not sp.limit(cancellation, R, 0).has(sp.zoo, sp.nan, sp.oo, -sp.oo))

    # Independent algebraic raw-vs-regular outer source comparison.  The
    # symbols represent analytic point values of J,J_T,J_R,K,Q,eth_6 K.
    J, JT, JR, K, Q, eth6K, eps0, epsbar0, tau0, pibar0, psi20 = sp.symbols(
        "J JT JR K Q eth6K eps0 epsbar0 tau0 pibar0 psi20"
    )
    mode = sp.symbols("m", integer=True)
    time = R * 2 * M * (2 * M - a**2 * R / L2) / D
    thorn5J = (
        time * JT + b * (R * JR + 5 * J) +
        R * I * a * mode * J / D - R * (3 * eps0 - epsbar0) * J
    )
    raw_over_r6 = (
        thorn5J - (4 * rho0 + rhobar0) * J + R * eth6K +
        R**2 * (-4 * tau0 + pibar0) * K - 3 * R * psi20 * Q
    )
    radial_regular = (
        2 * M * (2 * M - a**2 * R / L2) * JT / D + b * JR +
        (cancellation + I * a * mode / D - 3 * eps0 + epsbar0) * J
    )
    regular_over_r7 = (
        radial_regular + eth6K + R * (-4 * tau0 + pibar0) * K -
        3 * psi20 * Q
    )
    check("raw S0/R6 equals R times regular S0/R7",
          sp.factor(sp.cancel(raw_over_r6 - R * regular_over_r7)) == 0)
    check("raw source has seventh-order peeling",
          sp.factor(sp.cancel(raw_over_r6 / R - regular_over_r7)) == 0)

    forcing = 2 * D * dm**4 * regular_over_r7
    check("correct forcing finite at scri",
          not sp.limit(forcing, R, 0).has(sp.zoo, sp.nan, sp.oo, -sp.oo))
    old_forcing_for_same_source = 2 * D * R**4 * regular_over_r7
    check("old forcing spuriously vanishes as R4",
          sp.limit(old_forcing_for_same_source / R**4, R, 0) ==
          2 * L2**2 * sp.limit(regular_over_r7, R, 0))

    # Deterministic substitutions cover Schwarzschild, moderate Kerr, and
    # near-extremal Kerr at interior and future-horizon coordinates.
    seeds = [
        {M: 1, a: 0, L2: sp.Rational(9, 5), y: sp.Rational(-2, 5), R: sp.Rational(1, 5)},
        {M: 1, a: sp.Rational(2, 3), L2: sp.Rational(6, 5), y: sp.Rational(1, 3), R: sp.Rational(3, 10)},
        {M: 1, a: sp.Rational(999, 1000), L2: 1, y: sp.Rational(-3, 7), R: sp.Rational(4, 5)},
    ]
    values = {
        J: sp.Rational(2, 7) + I * sp.Rational(1, 11),
        JT: -sp.Rational(3, 8) + I * sp.Rational(2, 9),
        JR: sp.Rational(5, 12) - I * sp.Rational(1, 6),
        K: -sp.Rational(1, 4) + I * sp.Rational(3, 10),
        Q: sp.Rational(7, 13) - I * sp.Rational(2, 15),
        eth6K: sp.Rational(1, 9) + I * sp.Rational(4, 17),
        eps0: sp.Rational(1, 20) - I * sp.Rational(1, 30),
        epsbar0: sp.Rational(1, 20) + I * sp.Rational(1, 30),
        tau0: sp.Rational(2, 25) + I * sp.Rational(1, 18),
        pibar0: -sp.Rational(1, 19) + I * sp.Rational(1, 23),
        psi20: -sp.Rational(3, 14) + I * sp.Rational(1, 27),
        mode: -2,
    }
    for index, point in enumerate(seeds):
        residual = sp.N((raw_over_r6 - R * regular_over_r7).subs(values).subs(point), 60)
        check(f"deterministic raw/compact point {index}", abs(complex(residual)) < 1e-50)

    # Fixed-seed rational random fields are independent substitutions into the
    # raw and cancellation-safe forms, not calls through production C++.
    generator = random.Random(0x5A17C0DE)
    for index in range(12):
        rational = lambda low, high: sp.Rational(generator.randint(low, high),
                                                  generator.randint(5, 31))
        point = {
            M: sp.Rational(1),
            a: rational(-9, 9),
            L2: rational(12, 40),
            y: rational(-4, 4),
            R: rational(1, 9),
        }
        field_values = dict(values)
        for symbol in (J, JT, JR, K, Q, eth6K, eps0, epsbar0, tau0,
                       pibar0, psi20):
            field_values[symbol] = rational(-10, 10) + I * rational(-10, 10)
        field_values[mode] = generator.choice((-4, -2, 0, 2, 4))
        residual = sp.cancel(
            (raw_over_r6 - R * regular_over_r7).subs(field_values).subs(point)
        )
        check(f"fixed-random raw/compact field {index}", residual == 0)

    scri_raw = sp.limit(raw_over_r6 / R, R, 0)
    scri_regular = sp.limit(regular_over_r7, R, 0)
    check("scri raw/compact limit equality",
          sp.simplify(scri_raw - scri_regular) == 0)


if __name__ == "__main__":
    main()
