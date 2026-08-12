#!/usr/bin/env python3
"""Exact foundation checks for the linear spin +2 companion derivation.

This is deliberately not a production source generator.  It checks the
convention conversions, field scaling, endpoint regularity, and the
normalization-independent part of the separated TSI.  Phase-sensitive TSI
tests remain blocked until normalized angular/radial mode fixtures exist.
"""

from __future__ import annotations

import sympy as sp


def require_zero(name: str, expression: sp.Expr) -> None:
    reduced = sp.factor(sp.cancel(sp.expand(expression)))
    if reduced != 0:
        raise AssertionError(f"{name}: expected zero, got {reduced}")
    print(f"PASS {name}")


def require_equal(name: str, left: sp.Expr, right: sp.Expr) -> None:
    require_zero(name, left - right)


def main() -> None:
    R, L, M, a, c = sp.symbols("R L M a c", nonzero=True)
    ell, m = sp.symbols("ell m", integer=True)
    r = L**2 / R
    angular_denominator = L**2 - sp.I * a * R * c
    zeta = r - sp.I * a * c

    # Ripley et al. use the continuous cube-root branch
    # (Psi2/M)^(1/3)=-zeta^(-1), hence its fourth power is zeta^(-4).
    psi2_over_m = -R**3 / angular_denominator**3
    selected_cube_root = -R / angular_denominator
    require_equal(
        "selected cube root cubes to Psi2/M",
        selected_cube_root**3,
        psi2_over_m,
    )
    w_plus = R * selected_cube_root**4
    require_equal(
        "plus field scaling is R zeta^-4",
        w_plus,
        R / zeta**4,
    )
    require_equal(
        "plus field scaling compact form",
        w_plus,
        R**5 / angular_denominator**4,
    )

    scri_leading = sp.limit(w_plus / R**5, R, 0)
    require_equal("scri leading W_plus/R^5", scri_leading, L**-8)

    r_plus = M + sp.sqrt(M**2 - a**2)
    r_h = L**2 / r_plus
    horizon_value = sp.factor(w_plus.subs(R, r_h))
    expected_horizon = sp.factor(
        r_h**5 / (L**2 - sp.I * a * r_h * c) ** 4
    )
    require_equal("finite horizon scaling expression", horizon_value, expected_horizon)

    # Tetrad boost/spin transformation: the product Psi0*Psi4 is invariant.
    boost, phase, psi0_k, psi4_k = sp.symbols(
        "boost phase psi0_k psi4_k", nonzero=True
    )
    psi0_code = boost**2 * phase**2 * psi0_k
    psi4_code = boost**-2 * phase**-2 * psi4_k
    require_equal(
        "extreme Weyl boost-spin product invariant",
        psi0_code * psi4_code,
        psi0_k * psi4_k,
    )

    # Dividing Ripley et al.'s explicit C10c tetrad by C7c gives
    # q=-(r+i a cos(theta))/(r-i a cos(theta))
    #   = exp[-2 i atan(r/(a cos(theta)))].  Printed Eq. C9c has sin(theta),
    # which is inconsistent with the two explicit tetrads and is a typo.
    # Test the consistent phase through y=r/(a cos(theta)).
    y = sp.symbols("y", real=True)
    phase_rational = (1 - sp.I * y) / (1 + sp.I * y)
    x, radial = sp.symbols("x radial", real=True, nonzero=True)
    explicit_tetrad_ratio = -(radial + sp.I * x) / (radial - sp.I * x)
    require_equal(
        "explicit tetrad phase rationalization",
        phase_rational.subs(y, radial / x),
        explicit_tetrad_ratio,
    )
    require_equal(
        "tetrad phase has unit norm",
        sp.simplify(phase_rational * sp.conjugate(phase_rational)),
        1,
    )

    # NP-to-GHP conversion for the two inner spin coefficients.
    D_hmm, D_hlm, delta_hll = sp.symbols("D_hmm D_hlm delta_hll")
    thorn_hmm, thorn_hlm, eth_hll = sp.symbols(
        "thorn_hmm thorn_hlm eth_hll"
    )
    epsilon, epsilon_bar, beta, alpha_bar = sp.symbols(
        "epsilon epsilon_bar beta alpha_bar"
    )
    rho, rho_bar, tau, pi_bar = sp.symbols("rho rho_bar tau pi_bar")
    hmm, hlm, hll = sp.symbols("hmm hlm hll")

    substitutions = {
        D_hmm: thorn_hmm + 2 * epsilon * hmm - 2 * epsilon_bar * hmm,
        D_hlm: thorn_hlm + 2 * epsilon * hlm,
        delta_hll: eth_hll + 2 * beta * hll + 2 * alpha_bar * hll,
    }
    sigma_np = (
        sp.Rational(1, 2)
        * (D_hmm + 2 * (epsilon_bar - epsilon) * hmm + (rho - rho_bar) * hmm)
        - (tau + pi_bar) * hlm
    )
    sigma_ghp = (
        sp.Rational(1, 2) * (thorn_hmm + (rho - rho_bar) * hmm)
        - (tau + pi_bar) * hlm
    )
    require_equal("sigma NP-to-GHP rewrite", sigma_np.subs(substitutions), sigma_ghp)

    kappa_np = (
        D_hlm
        - (2 * epsilon + rho_bar) * hlm
        - sp.Rational(1, 2)
        * (
            delta_hll
            - 2 * alpha_bar * hll
            - 2 * beta * hll
            + (pi_bar + tau) * hll
        )
    )
    kappa_ghp = (
        thorn_hlm
        - rho_bar * hlm
        - sp.Rational(1, 2) * (eth_hll + (pi_bar + tau) * hll)
    )
    require_equal("kappa NP-to-GHP rewrite", kappa_np.subs(substitutions), kappa_ghp)

    # For kappa, (s,b)=(1,2), hence (p,q)=(3,1).
    delta_kappa, eth_kappa, kappa = sp.symbols(
        "delta_kappa eth_kappa kappa"
    )
    # The delta-to-eth equality is definitional; check its insertion into the
    # NP outer operator rather than equating independent symbols directly.
    outer_np = delta_kappa + (tau - pi_bar + alpha_bar + 3 * beta) * kappa
    outer_in_eth = eth_kappa + (tau - pi_bar + 2 * alpha_bar + 6 * beta) * kappa
    require_equal(
        "outer Psi0 angular operator NP-to-eth rewrite",
        outer_np.subs({delta_kappa: eth_kappa + 3 * beta * kappa + alpha_bar * kappa}),
        outer_in_eth,
    )

    # Spin/boost bookkeeping for T0.
    def weight_add(*weights: tuple[int, int]) -> tuple[int, int]:
        return tuple(sum(w[i] for w in weights) for i in range(2))  # type: ignore[return-value]

    assert weight_add((2, 1), (0, 1)) == (2, 2)  # thorn sigma
    assert weight_add((1, 2), (1, 0)) == (2, 2)  # delta kappa
    print("PASS T0 spin/boost weights")

    # Schwarzschild angular TSI product from Berens et al. Eq. (3.17).
    lambda_plus = ell * (ell + 1) - 6
    d_schwarzschild = (lambda_plus + 4) ** 2 * (lambda_plus + 6) ** 2
    d_expected = ((ell - 1) * ell * (ell + 1) * (ell + 2)) ** 2
    require_equal("Schwarzschild angular Starobinsky product", d_schwarzschild, d_expected)
    require_equal("ell=2 angular Starobinsky product", d_schwarzschild.subs(ell, 2), 24**2)

    omega = sp.symbols("omega")
    c_radial = d_schwarzschild + (12 * M * omega) ** 2
    require_equal(
        "radial-angular Starobinsky product offset",
        c_radial - d_schwarzschild,
        (12 * M * omega) ** 2,
    )

    # Compact spin +2 radial coefficients are finite polynomials at scri and
    # the horizon.  H_R has the expected two characteristic zeros.
    h_radial = R**2 * (L**4 - 2 * L**2 * M * R + a**2 * R**2) / L**4
    require_equal("H_R vanishes at scri", sp.limit(h_radial, R, 0), 0)
    require_equal("H_R vanishes at outer horizon", h_radial.subs(R, r_h), 0)

    s = sp.Integer(2)
    q_plus = 2 * R * (
        1 + s - (3 + s) * M * R / L**2 + 2 * a**2 * R**2 / L**4
    ) - 2 * sp.I * a * m * R**2 / L**2
    psi_plus = -2 * R * (
        (1 + s) * M / L**2 - a**2 * R / L**4
    ) - 2 * sp.I * a * m * R / L**2
    assert not sp.denom(sp.together(q_plus)).has(R)
    assert not sp.denom(sp.together(psi_plus)).has(R)
    require_equal("spin +2 Q coefficient scri limit", sp.limit(q_plus, R, 0), 0)
    require_equal(
        "spin +2 field coefficient scri limit", sp.limit(psi_plus, R, 0), 0
    )
    print("PASS compact spin +2 lower-order endpoint finiteness")

    print("Completed linear spin +2 foundation checks")


if __name__ == "__main__":
    main()
