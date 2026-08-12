#!/usr/bin/env python3
"""Independent Kerr-background derivative audit for the Route-B source.

The production implementation uses a small C++ coordinate jet.  This oracle
instead lets SymPy differentiate the printed rotated-Kinnersley expressions,
then evaluates them with mpmath at 80 digits.  It covers Schwarzschild,
positive/negative spin, scri, the future horizon, and near-axis points.
"""

from __future__ import annotations

import mpmath as mp
import sympy as sp


R, y = sp.symbols("R y", real=True)
M, a, L = sp.symbols("M a L", real=True, nonzero=True)
I = sp.I
sqrt2 = sp.sqrt(2)
s = sp.sqrt(1 - y**2)
dm = L**2 - I * a * R * y
dp = L**2 + I * a * R * y

beta0 = (-L**2 * y / s + I * a * R * s * (1 / s**2 + 1)) / (
    2 * sqrt2 * dm**2
)
epsilon0 = (
    L**2 * M - a**2 * R - I * a * (L**2 - M * R) * y
) / (2 * dm**2 * dp)

# Apply the coordinate tetrad vectors to the physical, rescaled coefficients.
definitions = (
    R**-1 * (R**2 / L**2) * sp.diff(R * beta0, R),
    R**-2 * (R**2 / L**2) * sp.diff(R**2 * epsilon0, R),
    R**-3
    * (R / (sqrt2 * dp) * s * sp.diff(R**2 * epsilon0, y)),
)

# Product-rule forms used as a second symbolic route and mirrored only at the
# final operator boundary in C++.
product_forms = (
    R / L**2 * (beta0 + R * sp.diff(beta0, R)),
    (2 * R * epsilon0 + R**2 * sp.diff(epsilon0, R)) / L**2,
    s * sp.diff(epsilon0, y) / (sqrt2 * dp),
)

for name, definition, product in zip(
    ("Delta1 beta0", "Delta2 epsilon0", "bardelta2 epsilon0"),
    definitions,
    product_forms,
):
    if sp.simplify(definition - product) != 0:
        raise AssertionError(f"product-rule mismatch for {name}")

# Exact Schwarzschild limits are useful independent normalization checks.
schwarzschild = tuple(sp.simplify(expr.subs(a, 0)) for expr in definitions)
expected_schwarzschild = (
    -R / (2 * sqrt2 * L**4) * y / s,
    M * R / L**6,
    sp.Integer(0),
)
if any(
    sp.simplify(actual - expected) != 0
    for actual, expected in zip(schwarzschild, expected_schwarzschild)
):
    raise AssertionError("Schwarzschild background-derivative limit mismatch")

mp.mp.dps = 80
# Evaluate the cancellation-safe product forms so the exact R=0 limit is
# represented directly rather than as a removable 0/0 in the definitions.
functions = [sp.lambdify((M, a, L, R, y), expr, "mpmath") for expr in product_forms]


def horizon_radius(mass: mp.mpf, spin: mp.mpf, length: mp.mpf) -> mp.mpf:
    return length**2 / (mass + mp.sqrt(mass**2 - spin**2))


cases = [
    (mp.mpf("1.0"), mp.mpf("0.0"), mp.mpf("1.0"), mp.mpf("0.0"), mp.mpf("0.2")),
    (mp.mpf("1.0"), mp.mpf("0.63"), mp.mpf("1.3"), mp.mpf("0.0"), mp.mpf("-0.41")),
    (mp.mpf("1.0"), mp.mpf("0.63"), mp.mpf("1.3"), horizon_radius(mp.mpf("1.0"), mp.mpf("0.63"), mp.mpf("1.3")), mp.mpf("0.37")),
    (mp.mpf("1.0"), mp.mpf("0.999"), mp.mpf("1.0"), horizon_radius(mp.mpf("1.0"), mp.mpf("0.999"), mp.mpf("1.0")), mp.mpf("0.999999999999")),
    (mp.mpf("1.2"), mp.mpf("-0.71"), mp.mpf("0.9"), mp.mpf("0.24"), mp.mpf("-0.999999999999")),
]

for case in cases:
    values = [function(*case) for function in functions]
    if not all(mp.isfinite(value.real) and mp.isfinite(value.imag) for value in values):
        raise AssertionError(f"non-finite background derivative at {case}")

# Radial falloff gate: all three rescaled slots are finite at scri.  The two
# Delta slots vanish there.  The angular epsilon derivative generally has a
# finite nonzero limit for rotating Kerr.
for spin in (mp.mpf("0"), mp.mpf("0.63"), mp.mpf("0.999"), mp.mpf("-0.71")):
    values = [function(mp.mpf("1"), spin, mp.mpf("1.1"), mp.mpf("0"), mp.mpf("0.3")) for function in functions]
    if abs(values[0]) > mp.mpf("1e-70") or abs(values[1]) > mp.mpf("1e-70"):
        raise AssertionError("Delta-rescaled background derivative does not vanish at scri")
    if not (mp.isfinite(values[2].real) and mp.isfinite(values[2].imag)):
        raise AssertionError("bardelta-rescaled background derivative is singular at scri")

print("plus2 stationary Kerr background derivative symbolic audit: PASS")
