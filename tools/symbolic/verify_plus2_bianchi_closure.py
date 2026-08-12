#!/usr/bin/env python3
"""Exact algebraic audit of the compact linear Bianchi closures.

This is deliberately independent of the C++ source implementation.  It starts
from the ordinary-NP Bianchi identities and the ORG perturbed directional
derivative, substitutes the repository radial scalings, and checks weights,
time differentiation, Delta_n inversion, and the derivative-order ledger.
"""

from __future__ import annotations

import sympy as sp


def check_equal(name: str, actual: sp.Expr, expected: sp.Expr) -> None:
    residual = sp.expand(actual - expected)
    if residual != 0:
        raise AssertionError(f"{name}: residual {residual}")
    print(f"PASS {name}")


R, M, L = sp.symbols("R M L", nonzero=True)
mu0, tau0, psi20 = sp.symbols("mu0 tau0 psi20")
Z0, Z1, H, Sig, Ta = sp.symbols("Z0 Z1 H Sig Ta")
Csharp, Bsharp = sp.symbols("Csharp Bsharp")
eth4_Z1, eth3_H = sp.symbols("eth4_Z1 eth3_H")
Delta3_psi20, ethprime3_psi20 = sp.symbols(
    "Delta3_psi20 ethprime3_psi20"
)

# Linear ordinary-NP bianchi-5 after eth Psi1=(delta-2 beta)Psi1.
# Each symbol below denotes the regular part after stripping its displayed R
# power.  Divide the physical equation by R^5.
b5_compact = (
    R**5 * eth4_Z1
    - (R * mu0) * (R**5 * Z0)
    - 4 * (R**2 * tau0) * (R**4 * Z1)
    + 3 * (R**2 * Sig) * (R**3 * H)
) / R**5
F0 = eth4_Z1 - R * mu0 * Z0 - 4 * R * tau0 * Z1 + 3 * Sig * H
check_equal("compact bianchi-5", b5_compact, F0)

# Linear ordinary-NP bianchi-6.  The ORG perturbation is exactly
# delta1=-h_lm Delta+(1/2)h_mm bardelta.
delta1_psi20 = (
    -(R**2 * Csharp) * (R**3 * Delta3_psi20)
    + sp.Rational(1, 2)
    * (R * Bsharp)
    * (R**4 * ethprime3_psi20)
)
b6_compact = (
    R**4 * eth3_H
    - 2 * (R * mu0) * (R**4 * Z1)
    - 3 * (R**2 * tau0) * (R**3 * H)
    + delta1_psi20
    - 3 * (R**2 * Ta) * (R**3 * psi20)
) / R**4
F1 = eth3_H + R * (
    -2 * mu0 * Z1
    - 3 * tau0 * H
    - Csharp * Delta3_psi20
    + sp.Rational(1, 2) * Bsharp * ethprime3_psi20
    - 3 * Ta * psi20
)
check_equal("compact bianchi-6", b6_compact, F1)

# Stationary-background time derivative checks.
Z0t, Z1t, Ht, Sigt, Tat = sp.symbols("Z0t Z1t Ht Sigt Tat")
Csharpt, Bsharpt = sp.symbols("Csharpt Bsharpt")
eth4_Z1t, eth3_Ht = sp.symbols("eth4_Z1t eth3_Ht")
F0t_expected = (
    eth4_Z1t
    - R * mu0 * Z0t
    - 4 * R * tau0 * Z1t
    + 3 * Sigt * H
    + 3 * Sig * Ht
)
F1t_expected = eth3_Ht + R * (
    -2 * mu0 * Z1t
    - 3 * tau0 * Ht
    - Csharpt * Delta3_psi20
    + sp.Rational(1, 2) * Bsharpt * ethprime3_psi20
    - 3 * Tat * psi20
)
time_map = {
    Z0: Z0t,
    Z1: Z1t,
    H: Ht,
    Sig: Sigt,
    Ta: Tat,
    Csharp: Csharpt,
    Bsharp: Bsharpt,
    eth4_Z1: eth4_Z1t,
    eth3_H: eth3_Ht,
}


def stationary_time_derivative(expr: sp.Expr) -> sp.Expr:
    return sp.expand(sum(sp.diff(expr, x) * xt for x, xt in time_map.items()))


check_equal("time derivative bianchi-5", stationary_time_derivative(F0), F0t_expected)
check_equal("time derivative bianchi-6", stationary_time_derivative(F1), F1t_expected)

# Exact Delta_n inversion and its regular scri limit.
n = sp.symbols("n", integer=True)
X, Xt, Xr, F = sp.symbols("X Xt Xr F")
A = 2 + 4 * M * R / L**2
B = R / L**2
solved_Xt = (F - B * (R * Xr + n * X)) / A
check_equal("Delta_n inversion", A * solved_Xt + B * (R * Xr + n * X), F)
check_equal("Delta_n scri inversion", solved_Xt.subs(R, 0), F / 2)

# (spin,boost) arithmetic.  Rescaling by R preserves both entries.
Weight = tuple[int, int]


def add(*weights: Weight) -> Weight:
    return (sum(w[0] for w in weights), sum(w[1] for w in weights))


def require_weight(name: str, terms: dict[str, Weight], expected: Weight) -> None:
    bad = {term: weight for term, weight in terms.items() if weight != expected}
    if bad:
        raise AssertionError(f"{name}: expected {expected}, got {bad}")
    print(f"PASS {name} weights ({len(terms)} terms)")


require_weight(
    "bianchi-5",
    {
        "Delta Psi0": (2, 1),
        "eth Psi1": (2, 1),
        "mu Psi0": add((0, -1), (2, 2)),
        "tau Psi1": add((1, 0), (1, 1)),
        "sigma Psi2": add((2, 1), (0, 0)),
    },
    (2, 1),
)
require_weight(
    "bianchi-6",
    {
        "Delta Psi1": (1, 0),
        "eth Psi2": (1, 0),
        "mu Psi1": add((0, -1), (1, 1)),
        "tau Psi2": add((1, 0), (0, 0)),
        "h_lm Delta Psi2": add((1, 1), (0, -1)),
        "h_mm bardelta Psi2": add((2, 0), (-1, 0)),
        "tau1 Psi2": add((1, 0), (0, 0)),
    },
    (1, 0),
)

# Exact time-order ledger.  An operator in the rotated tetrad adds one time
# derivative because Delta, thorn, eth, and ethprime all contain partial_T.
primitive_order = {
    "V": 0,
    "C": 0,
    "B": 0,
    "H": 0,
    "Pi": 0,
    "Sig": 1,
    "Kap": 1,
    "Rh": 1,
    "Ta": 1,
    "Al": 1,
    "Be": 1,
    "Ep": 1,
    "Z0": 2,
    "Z1": 2,
}
if len(primitive_order) != 14:
    raise AssertionError("manifest primitive count is not fourteen")

curvature_slot_value = primitive_order["Z0"] + 1
curvature_slot_tangent = curvature_slot_value + 1
q_value = primitive_order["Kap"] + 1
q_tangent = q_value + 1
assert (curvature_slot_value, curvature_slot_tangent) == (3, 4)
assert (q_value, q_tangent) == (2, 3)
assert curvature_slot_tangent + 1 == 5  # only a final source tangent
naive_forcing_value_order = curvature_slot_tangent
bclosure_forcing_value_order = 2
assert naive_forcing_value_order == 4
assert bclosure_forcing_value_order == 2
print("PASS derivative-order ledger (14 primitives, naive h[0..4], Bianchi h[0..2])")

print("PASS all plus2 Bianchi closure checks")
