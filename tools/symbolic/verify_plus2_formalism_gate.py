#!/usr/bin/env python3
"""Small independent checks for the spin +2 formalism gate.

This is not a source oracle.  It checks the discrete prime map, GHP weights,
the direct ORG operator substitutions, and the regular-field boundary factors
recorded in docs/PLUS2_FORMALISM_GATE.md.
"""

from __future__ import annotations

import sympy as sp


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    # Prime is an involution when the conventional minus signs are included.
    prime = {
        "kappa": ("nu", -1),
        "nu": ("kappa", -1),
        "sigma": ("lambda", -1),
        "lambda": ("sigma", -1),
        "rho": ("mu", -1),
        "mu": ("rho", -1),
        "tau": ("pi", -1),
        "pi": ("tau", -1),
        "beta": ("alpha", -1),
        "alpha": ("beta", -1),
        "epsilon": ("gamma", -1),
        "gamma": ("epsilon", -1),
    }
    for quantity, (image, sign) in prime.items():
        image2, sign2 = prime[image]
        check(f"prime involution: {quantity}", image2 == quantity and sign * sign2 == 1)

    # Ordinary-NP prime of d4.  In particular gamma'=-epsilon; gamma=0 in
    # the current rotated tetrad does not imply epsilon=0 after priming.
    D0, Delta0 = sp.symbols("D0 Delta0")
    rho0, mu0, eps0, gamma0 = sp.symbols("rho0 mu0 eps0 gamma0")
    rhobar0, mubar0, epsbar0, gammabar0 = sp.symbols("rhobar0 mubar0 epsbar0 gammabar0")
    d4 = Delta0 + 4 * mu0 + mubar0 + 3 * gamma0 - gammabar0
    prime_substitution = {
        Delta0: D0,
        mu0: -rho0,
        mubar0: -rhobar0,
        gamma0: -eps0,
        gammabar0: -epsbar0,
    }
    d4p = sp.expand(d4.xreplace(prime_substitution))
    check("ordinary-NP d4 prime includes epsilon", d4p == D0 - 4 * rho0 - rhobar0 - 3 * eps0 + epsbar0)

    # Formal sign ledger: prime the complete ungauge-fixed raw CL source,
    # including the two conventional minus signs lambda'=-sigma and
    # nu'=-kappa, before setting any ORG quantity to zero.
    names = "d4 d3 A B C Dop P4 P3 P2 P20 lam nu d41 d31 mu1 pi1"
    d4s, d3s, A, B, C, Dop, P4, P3, P2, P20, lam, nu, d41, d31, mu1, pi1 = sp.symbols(names)
    names_p = "d4p d3p A0 B0 C1 D1 P0 P1 sig kap d4p1 d3p1 rho1 tau1"
    d4ps, d3ps, A0, B0, C1, D1, P0, P1, sig, kap, d4p1, d3p1, rho1s, tau1s = sp.symbols(names_p)
    source4 = (
        -(d4s * A - d3s * B) * P4
        +(d4s * C - d3s * Dop) * P3
        -3 * (d4s * lam - d3s * nu) * P2
        +3 * P20 * ((d41 - 3 * mu1) * lam - (d31 - 3 * pi1) * nu)
    )
    prime_source = sp.expand(source4.xreplace({
        d4s: d4ps, d3s: d3ps, A: A0, B: B0, C: C1, Dop: D1,
        P4: P0, P3: P1, lam: -sig, nu: -kap, d41: d4p1,
        d31: d3p1, mu1: -rho1s, pi1: -tau1s,
    }))
    expected_source0 = sp.expand(
        -(d4ps * A0 - d3ps * B0) * P0
        +(d4ps * C1 - d3ps * D1) * P1
        +3 * (d4ps * sig - d3ps * kap) * P2
        -3 * P20 * ((d4p1 + 3 * rho1s) * sig - (d3p1 + 3 * tau1s) * kap)
    )
    check("ungauge-fixed raw source prime sign ledger", sp.simplify(prime_source - expected_source0) == 0)
    direct_org = sp.expand(expected_source0.subs(A0, 0))
    grouped_org = sp.expand(
        d3ps * B0 * P0 + d4ps * C1 * P1 - d3ps * D1 * P1
        +3 * d4ps * sig * P2 - 3 * d3ps * kap * P2
        -3 * P20 * ((d4p1 + 3 * rho1s) * sig - (d3p1 + 3 * tau1s) * kap)
    )
    check("direct ORG grouped source sign ledger", sp.simplify(direct_org - grouped_org) == 0)

    # (spin, boost) bookkeeping for every grouped source contribution.
    # D raises boost by one; delta raises spin by one.
    weights = {
        "Psi0": (2, 2),
        "Psi1": (1, 1),
        "Psi2": (0, 0),
        "sigma": (2, 1),
        "kappa": (1, 2),
        "B0Psi0": (1, 2),
        "D1Psi1": (1, 2),
        "C1Psi1": (2, 1),
    }
    d4_inner = [weights["C1Psi1"], tuple(sum(x) for x in zip(weights["sigma"], weights["Psi2"]))]
    d3_inner = [weights["B0Psi0"], weights["D1Psi1"], tuple(sum(x) for x in zip(weights["kappa"], weights["Psi2"]))]
    check("d4p inner weights", all(w == (2, 1) for w in d4_inner))
    check("d3p inner weights", all(w == (1, 2) for w in d3_inner))
    check("outer source weight", (d4_inner[0][0], d4_inner[0][1] + 1) == (2, 2))
    check("transverse source weight", (d3_inner[0][0] + 1, d3_inner[0][1]) == (2, 2))

    # Direct substitution of the Appendix-B ORG spin coefficients.
    Delta, delta, bardelta = sp.symbols("Delta delta bardelta")
    mu, mubar, pi, pibar, tau, taubar = sp.symbols("mu mubar pi pibar tau taubar")
    alpha, alphabar, beta, betabar = sp.symbols("alpha alphabar beta betabar")
    hll, hlm, hlbar, hmm, hbarbar = sp.symbols("hll hlm hlbar hmm hbarbar")

    alpha1 = -sp.Rational(1, 4) * (Delta - 2 * mu + mubar) * hlbar - sp.Rational(1, 4) * (delta - 2 * alphabar + pibar + tau) * hbarbar
    beta1 = -sp.Rational(1, 4) * (Delta + mu + 2 * mubar) * hlm + sp.Rational(1, 4) * (bardelta + 2 * betabar - pi - taubar) * hmm
    pi1 = -sp.Rational(1, 2) * (Delta + mubar) * hlbar - sp.Rational(1, 2) * tau * hbarbar
    tau1 = sp.Rational(1, 2) * (Delta + mu) * hlm - sp.Rational(1, 2) * pi * hmm
    epsilon1 = -sp.Rational(1, 4) * (Delta - mu + mubar) * hll - sp.Rational(1, 4) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar + sp.Rational(1, 4) * (bardelta - 2 * alpha - 3 * pi - 2 * taubar) * hlm
    rho1 = sp.Rational(1, 2) * mu * hll + sp.Rational(1, 2) * (bardelta - 2 * alpha - pi) * hlm - sp.Rational(1, 2) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar

    B0_coefficient = sp.expand(-4 * alpha1 + pi1)
    B0_expected = sp.expand((sp.Rational(1, 2) * Delta - 2 * mu + sp.Rational(1, 2) * mubar) * hlbar + (delta - 2 * alphabar + pibar + sp.Rational(1, 2) * tau) * hbarbar)
    check("ORG B0 scalar coefficient", sp.simplify(B0_coefficient - B0_expected) == 0)

    C1_coefficient = sp.expand(-2 * beta1 - 4 * tau1)
    C1_expected = sp.expand((-sp.Rational(3, 2) * Delta - sp.Rational(3, 2) * mu + mubar) * hlm + (-sp.Rational(1, 2) * bardelta - betabar + sp.Rational(5, 2) * pi + sp.Rational(1, 2) * taubar) * hmm)
    check("ORG C1 scalar coefficient", sp.simplify(C1_coefficient - C1_expected) == 0)

    D1_coefficient = sp.expand(-2 * epsilon1 - 4 * rho1)
    D1_expected = sp.expand((sp.Rational(1, 2) * Delta - sp.Rational(5, 2) * mu + sp.Rational(1, 2) * mubar) * hll + sp.Rational(5, 2) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar + (-sp.Rational(5, 2) * bardelta + 5 * alpha + sp.Rational(7, 2) * pi + taubar) * hlm)
    check("ORG D1 scalar coefficient", sp.simplify(D1_coefficient - D1_expected) == 0)

    # W_plus and endpoint coefficient regularity.
    R, L, a, y, RH = sp.symbols("R L a y RH", positive=True, finite=True, real=True)
    I = sp.I
    W = R**5 / (L**2 - I * a * R * y) ** 4
    check("W_plus scri order five", sp.limit(W / R**5, R, 0) == L**-8)
    WH = sp.simplify(W.subs(R, RH))
    check("W_plus horizon finite expression", not WH.has(sp.zoo, sp.nan, sp.oo, -sp.oo))
    denom_abs_sq = sp.expand((L**2 - I * a * RH * y) * (L**2 + I * a * RH * y))
    check("W_plus denominator nonzero", denom_abs_sq == L**4 + a**2 * RH**2 * y**2)

    # Ripley substitutes W_s inside O_s[W_s Z_s] and then multiplies the NP
    # equation by N=2 Sigma/R.  The physical RHS normalization is identical
    # for both signs; there is no extra division by W_s.
    S = sp.symbols("S")
    Sigma_bl = (L**4 + a**2 * R**2 * y**2) / R**2
    forcing_np = sp.factor((2 * Sigma_bl / R) * S)
    forcing_expected = 2 * (L**4 + a**2 * R**2 * y**2) * S / R**3
    check("common coordinate source normalization", sp.simplify(forcing_np - forcing_expected) == 0)
    check("source normalization independent of W_plus", not forcing_np.has(W))

    # Generic Teukolsky angular eigenvalue for s=+2.
    ell = sp.symbols("ell", integer=True, nonnegative=True)
    eig = -(ell - 2) * (ell + 3)
    check("s=+2 ell=2 angular eigenvalue", eig.subs(ell, 2) == 0)
    check("s=+2 ell=3 angular eigenvalue", eig.subs(ell, 3) == -6)


if __name__ == "__main__":
    main()
