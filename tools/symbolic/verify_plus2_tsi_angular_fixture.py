#!/usr/bin/env python3
"""Normalization-pinned Schwarzschild angular TSI fixture.

This checks the portion of the separated Teukolsky--Starobinsky fixture that
does not require numerical confluent-Heun radial modes.  Every normalization
factor comes from Berens--Gravely--Lupsasca arXiv:2403.20311 Eqs. (2.6),
(2.22)--(2.29), (2.37), and (2.44), as also transcribed in their supplemental
ExampleUsage.nb at commit cf924707593a58ec889c70ea501d764e99d1d4aa.
"""

from __future__ import annotations

import sympy as sp


def require_equal(name: str, left: sp.Expr, right: sp.Expr) -> None:
    residual = sp.factor(
        sp.cancel(sp.together(sp.trigsimp(sp.expand_trig(left - right))))
    )
    if residual != 0:
        raise AssertionError(f"{name}: expected zero residual, got {residual}")
    print(f"PASS {name}")


def main() -> None:
    theta = sp.symbols("theta", real=True)
    m = sp.Integer(2)

    # Eq. (2.6), a*omega=0, ell=m=2.  The HeunC factors have q=0 and
    # H(0)=1, reducing identically to one in this lowest mode.
    s_plus_hat = (1 - sp.cos(theta)) ** 2
    s_minus_hat = (1 + sp.cos(theta)) ** 2
    require_equal(
        "hatted plus angular mode has Eq2.6 north-pole normalization",
        sp.limit(s_plus_hat / (1 - sp.cos(theta)) ** 2, theta, 0),
        1,
    )
    require_equal(
        "hatted minus angular mode has Eq2.6 north-pole normalization",
        sp.limit(s_minus_hat, theta, 0),
        4,
    )

    # Eq. (2.32) with a*omega=0.
    def angular_l(n: int, value: sp.Expr) -> sp.Expr:
        return (
            sp.diff(value, theta)
            + m / sp.sin(theta) * value
            + n * sp.cot(theta) * value
        )

    def angular_l_dagger(n: int, value: sp.Expr) -> sp.Expr:
        return (
            sp.diff(value, theta)
            - m / sp.sin(theta) * value
            + n * sp.cot(theta) * value
        )

    lambda_plus = sp.Integer(0)
    d_product = (lambda_plus + 4) ** 2 * (lambda_plus + 6) ** 2
    d_hat = sp.factorial(m - 2) / sp.factorial(m + 2) * d_product
    d_hat_prime = sp.factorial(m + 2) / sp.factorial(m - 2)
    require_equal(
        "hatted angular constants multiply to D", d_hat * d_hat_prime, d_product
    )
    require_equal("ell2 m2 Dhat is normalization-pinned", d_hat, 24)
    require_equal("ell2 m2 DhatPrime is normalization-pinned", d_hat_prime, 24)

    raised = s_plus_hat
    for n in (2, 1, 0, -1):
        raised = sp.trigsimp(angular_l(n, raised))
    require_equal(
        "ATSI1 Eq2.37a for normalized ell2 m2 modes",
        raised,
        d_hat * s_minus_hat,
    )

    lowered = s_minus_hat
    for n in (2, 1, 0, -1):
        lowered = sp.trigsimp(angular_l_dagger(n, lowered))
    require_equal(
        "ATSI1 Eq2.37b for normalized ell2 m2 modes",
        lowered,
        d_hat_prime * s_plus_hat,
    )

    # Candidate full-fixture metadata: Schwarzschild M=1, real omega=1/5,
    # ell=m=2, horizon-in modes.  Eqs. (2.27)--(2.29) pin the radial factors;
    # this algebra does not pretend to evaluate the normalized HeunC modes.
    mass = sp.Integer(1)
    omega = sp.Rational(1, 5)
    r_plus = sp.Integer(2)
    r_minus = sp.Integer(0)
    sigma = r_plus - r_minus
    w = 4 * mass * omega * r_plus
    gamma = (w + 2 * sp.I * sigma) * (w + sp.I * sigma) * w * (w - sp.I * sigma)
    c_product = d_product + (12 * mass * omega) ** 2
    c_hat_in = gamma
    c_hat_in_prime = c_product / gamma
    require_equal(
        "hatted horizon-in radial constants multiply to C",
        c_hat_in * c_hat_in_prime,
        c_product,
    )

    same_mode_amplitude = 4 * d_hat_prime / c_hat_in_prime
    # Eq. (2.44) conjugates the hatted radial factor in the sharp sector.
    sharp_mode_amplitude = 48 * sp.I * omega * mass / sp.conjugate(c_hat_in_prime)
    require_equal(
        "Eq2.44 signed sharp-to-same phase ratio is pinned",
        sharp_mode_amplitude / same_mode_amplitude,
        sp.I / 10 * c_hat_in_prime / sp.conjugate(c_hat_in_prime),
    )
    if sp.simplify(sharp_mode_amplitude / same_mode_amplitude - sp.I / 10) == 0:
        raise AssertionError("complex radial phase was accidentally discarded")
    print("PASS Eq2.44 ratio is not the phase-free i/10 shortcut")

    print("Completed normalization-pinned angular TSI fixture checks")
    print(
        "PASS angular subfixture; normalized radial modes are checked by the numerical gate"
    )


if __name__ == "__main__":
    main()
