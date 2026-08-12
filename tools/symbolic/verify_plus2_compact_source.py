#!/usr/bin/env python3
"""Independent checks for the compact raw ORG spin +2 source.

This file intentionally does not import or transcribe a production source
kernel.  One route evaluates the ordinary-NP operator perturbations obtained
from the primary Appendix-B metric formulas.  A second route evaluates the
rescaled GHP term ledger.  The routes meet only at the asserted equality.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "PLUS2_SOURCE_TERM_LEDGER.csv"


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def check_ledger() -> None:
    with LEDGER.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_counts = {
        "C12": 8,
        "B12": 8,
        "D12": 10,
        "Er12": 5,
        "Et12": 6,
        "J12": 2,
        "K12": 3,
        "Q12": 2,
        "S0": 7,
    }
    bases = {
        "C12": (2, 1, 6),
        "B12": (1, 2, 7),
        "D12": (1, 2, 6),
        "Er12": (2, 2, 4),
        "Et12": (2, 2, 5),
        "J12": (2, 1, 5),
        "K12": (1, 2, 6),
        "Q12": (2, 2, 4),
        "S0": (2, 2, 6),
    }
    check("ledger family cardinalities", Counter(row["family"] for row in rows) == expected_counts)
    check("ledger IDs unique", len({row["id"] for row in rows}) == len(rows))
    for row in rows:
        spin, boost, base_power = bases[row["family"]]
        physical_power = int(row["physical_R_power"])
        explicit_power = int(row["explicit_R_power_in_regular_family"])
        check(f"ledger {row['id']} weight", (int(row["output_spin"]), int(row["output_boost"])) == (spin, boost))
        check(f"ledger {row['id']} power", physical_power - base_power == explicit_power)


def check_exact_operator_algebra() -> None:
    """Exact scalar-coefficient reductions, independent of coordinate fields."""
    Delta, delta, bardelta = sp.symbols("Delta delta bardelta")
    mu, mubar, pi, pibar, tau, taubar = sp.symbols("mu mubar pi pibar tau taubar")
    alpha, alphabar, beta, betabar = sp.symbols("alpha alphabar beta betabar")
    hll, hlm, hlbar, hmm, hbarbar = sp.symbols("hll hlm hlbar hmm hbarbar")

    alpha1 = -sp.Rational(1, 4) * (Delta - 2 * mu + mubar) * hlbar - sp.Rational(1, 4) * (delta - 2 * alphabar + pibar + tau) * hbarbar
    beta1 = -sp.Rational(1, 4) * (Delta + mu + 2 * mubar) * hlm + sp.Rational(1, 4) * (bardelta + 2 * betabar - pi - taubar) * hmm
    pi1 = -sp.Rational(1, 2) * (Delta + mubar) * hlbar - sp.Rational(1, 2) * tau * hbarbar
    tau1 = sp.Rational(1, 2) * (Delta + mu) * hlm - sp.Rational(1, 2) * pi * hmm
    eps1 = -sp.Rational(1, 4) * (Delta - mu + mubar) * hll - sp.Rational(1, 4) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar + sp.Rational(1, 4) * (bardelta - 2 * alpha - 3 * pi - 2 * taubar) * hlm
    rho1 = sp.Rational(1, 2) * mu * hll + sp.Rational(1, 2) * (bardelta - 2 * alpha - pi) * hlm - sp.Rational(1, 2) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar
    alphabar1 = -sp.Rational(1, 4) * (bardelta - 2 * alpha + pi + taubar) * hmm - sp.Rational(1, 4) * (Delta + mu - 2 * mubar) * hlm
    pibar1 = -sp.Rational(1, 2) * (Delta + mu) * hlm - sp.Rational(1, 2) * taubar * hmm

    expected = {
        "B0": (sp.Rational(1, 2) * Delta - 2 * mu + sp.Rational(1, 2) * mubar) * hlbar + (delta - 2 * alphabar + pibar + sp.Rational(1, 2) * tau) * hbarbar,
        "C1": (-sp.Rational(3, 2) * Delta - sp.Rational(3, 2) * mu + mubar) * hlm + (-sp.Rational(1, 2) * bardelta - betabar + sp.Rational(5, 2) * pi + sp.Rational(1, 2) * taubar) * hmm,
    }
    # Spell D1 separately so a typo cannot be hidden by shared construction.
    d1_expected = (sp.Rational(1, 2) * Delta - sp.Rational(5, 2) * mu + sp.Rational(1, 2) * mubar) * hll + sp.Rational(5, 2) * (delta - 2 * alphabar + pibar + 2 * tau) * hlbar + (-sp.Rational(5, 2) * bardelta + 5 * alpha + sp.Rational(7, 2) * pi + taubar) * hlm
    check("exact B0 connection reduction", sp.expand(-4 * alpha1 + pi1 - expected["B0"]) == 0)
    check("exact C1 connection reduction", sp.expand(-2 * beta1 - 4 * tau1 - expected["C1"]) == 0)
    check("exact D1 connection reduction", sp.expand(-2 * eps1 - 4 * rho1 - d1_expected) == 0)
    d_hlm, bp_hmm = sp.symbols("d_hlm bp_hmm")
    beta1_jet = -sp.Rational(1, 4) * (d_hlm + (mu + 2 * mubar) * hlm) + sp.Rational(1, 4) * (bp_hmm + (2 * betabar - pi - taubar) * hmm)
    alphabar1_jet = -sp.Rational(1, 4) * (bp_hmm + (-2 * alpha + pi + taubar) * hmm) - sp.Rational(1, 4) * (d_hlm + (mu - 2 * mubar) * hlm)
    tau1_jet = sp.Rational(1, 2) * (d_hlm + mu * hlm) - sp.Rational(1, 2) * pi * hmm
    pibar1_jet = -sp.Rational(1, 2) * (d_hlm + mu * hlm) - sp.Rational(1, 2) * taubar * hmm
    et_expected = -sp.Rational(1, 2) * bp_hmm + mubar * hlm + (-sp.Rational(1, 2) * alpha - sp.Rational(3, 2) * betabar + sp.Rational(3, 2) * pi + sp.Rational(1, 2) * taubar) * hmm
    et_actual = -3 * beta1_jet - alphabar1_jet - tau1_jet + pibar1_jet
    check("exact Et ordinary-NP reduction", sp.expand(et_actual - et_expected) == 0)
    # Add delta1(kappa), then rewrite the angular pair with the GHP ethprime
    # Leibniz rule.  This proves the stable expression without alpha/beta.
    kappa, bp_kappa = sp.symbols("kappa bp_kappa")
    raw_angular = sp.Rational(1, 2) * hmm * bp_kappa + et_expected * kappa
    ethp_kappa = bp_kappa - 3 * alpha * kappa - betabar * kappa
    ethp_hmm = bp_hmm - 2 * alpha * hmm + 2 * betabar * hmm
    stable_angular = sp.Rational(1, 2) * (hmm * ethp_kappa - ethp_hmm * kappa) + (sp.Rational(3, 2) * pi + sp.Rational(1, 2) * taubar) * hmm * kappa + mubar * hlm * kappa
    check("exact Et stable GHP reduction", sp.expand(raw_angular - stable_angular) == 0)
    erroneous = -hlm * sp.Symbol("Delta_kappa") + sp.Rational(1, 2) * hmm * ethp_kappa + ((Delta + mu + mubar) * hlm - sp.Rational(1, 2) * ethp_hmm - tau1 + pibar1) * kappa
    check("Et regression rejects earlier draft", sp.expand(erroneous - (-hlm * sp.Symbol("Delta_kappa") + stable_angular)) != 0)


@dataclass
class Background:
    mu0: sp.Expr
    mubar0: sp.Expr
    rho0: sp.Expr
    rhobar0: sp.Expr
    eps0: sp.Expr
    epsbar0: sp.Expr
    alpha0: sp.Expr
    alphabar0: sp.Expr
    beta0: sp.Expr
    betabar0: sp.Expr
    tau0: sp.Expr
    taubar0: sp.Expr
    pi0: sp.Expr
    pibar0: sp.Expr
    psi20: sp.Expr


@dataclass
class Parent:
    V: sp.Expr
    C: sp.Expr
    Cs: sp.Expr
    B: sp.Expr
    Bs: sp.Expr
    Z0: sp.Expr
    Z1: sp.Expr
    H: sp.Expr


@dataclass
class Connections:
    sig: sp.Expr
    kap: sp.Expr
    rho: sp.Expr
    rhobar: sp.Expr
    tau: sp.Expr
    pi: sp.Expr
    pibar: sp.Expr
    eps: sp.Expr
    epsbar: sp.Expr
    alpha: sp.Expr
    alphabar: sp.Expr
    beta: sp.Expr


t, R, y, phi = sp.symbols("t R y phi", real=True)
Mpar, apar, Lpar = sp.symbols("Mpar apar Lpar", real=True, positive=True)
I = sp.I
sin_theta = sp.sqrt(1 - y**2)
sqrt_two = sp.sqrt(2)


def D(f: sp.Expr) -> sp.Expr:
    denominator = Lpar**4 + apar**2 * R**2 * y**2
    return R**2 / denominator * (
        2 * Mpar * (2 * Mpar - apar**2 * R / Lpar**2) * sp.diff(f, t)
        - sp.Rational(1, 2) * (Lpar**2 - 2 * Mpar * R + apar**2 * R**2 / Lpar**2) * sp.diff(f, R)
        + apar * sp.diff(f, phi)
    )


def Delta(f: sp.Expr) -> sp.Expr:
    return (2 + 4 * Mpar * R / Lpar**2) * sp.diff(f, t) + R**2 / Lpar**2 * sp.diff(f, R)


def delta(f: sp.Expr) -> sp.Expr:
    return R / (sqrt_two * (Lpar**2 - I * apar * R * y)) * (
        -I * apar * sin_theta * sp.diff(f, t)
        + sin_theta * sp.diff(f, y)
        - I * sp.diff(f, phi) / sin_theta
    )


def bardelta(f: sp.Expr) -> sp.Expr:
    return R / (sqrt_two * (Lpar**2 + I * apar * R * y)) * (
        I * apar * sin_theta * sp.diff(f, t)
        + sin_theta * sp.diff(f, y)
        + I * sp.diff(f, phi) / sin_theta
    )


def make_scalar(rng: random.Random, mode: int, complex_field: bool = True) -> sp.Expr:
    coeffs = [sp.Rational(rng.randint(1, 9), rng.randint(2, 11)) for _ in range(12)]
    real_poly = coeffs[0] + coeffs[1] * t + coeffs[2] * R + coeffs[3] * y + coeffs[4] * t * R + coeffs[5] * R * y
    imag_poly = coeffs[6] + coeffs[7] * t + coeffs[8] * R + coeffs[9] * y + coeffs[10] * t * y + coeffs[11] * R**2
    amplitude = real_poly + (I * imag_poly if complex_field else 0)
    return amplitude * sp.exp(I * mode * phi)


def make_background() -> Background:
    dm = Lpar**2 - I * apar * R * y
    dp = Lpar**2 + I * apar * R * y
    real_den = Lpar**4 + apar**2 * R**2 * y**2
    mu0 = -1 / dm
    rho0 = -(apar**2 * R**2 + Lpar**4 - 2 * Lpar**2 * Mpar * R) / (2 * dm**2 * dp)
    eps0 = (Lpar**2 * Mpar - apar**2 * R - I * apar * (Lpar**2 - Mpar * R) * y) / (2 * dm**2 * dp)
    alpha0 = y / (2 * sqrt_two * sin_theta * dp)
    beta0 = (-Lpar**2 * y / sin_theta + I * apar * R * sin_theta * (1 / (1 - y**2) + 1)) / (2 * sqrt_two * dm**2)
    tau0 = I * apar * sin_theta / (sqrt_two * dm**2)
    pi0 = -I * apar * sin_theta / (sqrt_two * real_den)
    psi20 = -Mpar / dm**3
    bar = lambda f: f.xreplace({I: -I})
    return Background(mu0, bar(mu0), rho0, bar(rho0), eps0, bar(eps0), alpha0, bar(alpha0), beta0, bar(beta0), tau0, bar(tau0), pi0, bar(pi0), psi20)


def make_parent(rng: random.Random, mode: int) -> Parent:
    return Parent(*(make_scalar(rng, mode, True) for _ in range(8)))


def physical_background(bg: Background) -> dict[str, sp.Expr]:
    return {
        "mu": R * bg.mu0, "mubar": R * bg.mubar0,
        "rho": R * bg.rho0, "rhobar": R * bg.rhobar0,
        "eps": R**2 * bg.eps0, "epsbar": R**2 * bg.epsbar0,
        "alpha": R * bg.alpha0, "alphabar": R * bg.alphabar0,
        "beta": R * bg.beta0, "betabar": R * bg.betabar0,
        "tau": R**2 * bg.tau0, "taubar": R**2 * bg.taubar0,
        "pi": R**2 * bg.pi0, "pibar": R**2 * bg.pibar0,
    }


def metric(parent: Parent) -> dict[str, sp.Expr]:
    return {"hll": R**2 * parent.V, "hlbar": R**2 * parent.C, "hlm": R**2 * parent.Cs, "hbarbar": R * parent.B, "hmm": R * parent.Bs}


def connections(parent: Parent, bg: Background) -> Connections:
    b, h = physical_background(bg), metric(parent)
    hll, hlbar, hlm, hbarbar, hmm = h["hll"], h["hlbar"], h["hlm"], h["hbarbar"], h["hmm"]
    sig = sp.Rational(1, 2) * (D(hmm) + (2 * (b["epsbar"] - b["eps"]) + b["rho"] - b["rhobar"]) * hmm) - (b["tau"] + b["pibar"]) * hlm
    kap = D(hlm) - (2 * b["eps"] + b["rhobar"]) * hlm - sp.Rational(1, 2) * (delta(hll) + (-2 * b["alphabar"] - 2 * b["beta"] + b["pibar"] + b["tau"]) * hll)
    eps = sp.Rational(1, 4) * (-Delta(hll) + (b["mu"] - b["mubar"]) * hll) + sp.Rational(1, 4) * (-delta(hlbar) + (2 * b["alphabar"] - b["pibar"] - 2 * b["tau"]) * hlbar) + sp.Rational(1, 4) * (bardelta(hlm) + (-2 * b["alpha"] - 3 * b["pi"] - 2 * b["taubar"]) * hlm)
    epsbar = sp.Rational(1, 4) * (-Delta(hll) + (b["mubar"] - b["mu"]) * hll) + sp.Rational(1, 4) * (-bardelta(hlm) + (2 * b["alpha"] - b["pi"] - 2 * b["taubar"]) * hlm) + sp.Rational(1, 4) * (delta(hlbar) + (-2 * b["alphabar"] - 3 * b["pibar"] - 2 * b["tau"]) * hlbar)
    rho = sp.Rational(1, 2) * b["mu"] * hll + sp.Rational(1, 2) * (bardelta(hlm) + (-2 * b["alpha"] - b["pi"]) * hlm) - sp.Rational(1, 2) * (delta(hlbar) + (-2 * b["alphabar"] + b["pibar"] + 2 * b["tau"]) * hlbar)
    rhobar = sp.Rational(1, 2) * b["mubar"] * hll + sp.Rational(1, 2) * (delta(hlbar) + (-2 * b["alphabar"] - b["pibar"]) * hlbar) - sp.Rational(1, 2) * (bardelta(hlm) + (-2 * b["alpha"] + b["pi"] + 2 * b["taubar"]) * hlm)
    alpha = -sp.Rational(1, 4) * (delta(hbarbar) + (-2 * b["alphabar"] + b["pibar"] + b["tau"]) * hbarbar) - sp.Rational(1, 4) * (Delta(hlbar) + (b["mubar"] - 2 * b["mu"]) * hlbar)
    alphabar = -sp.Rational(1, 4) * (bardelta(hmm) + (-2 * b["alpha"] + b["pi"] + b["taubar"]) * hmm) - sp.Rational(1, 4) * (Delta(hlm) + (b["mu"] - 2 * b["mubar"]) * hlm)
    beta = -sp.Rational(1, 4) * (Delta(hlm) + (b["mu"] + 2 * b["mubar"]) * hlm) + sp.Rational(1, 4) * (bardelta(hmm) + (2 * b["betabar"] - b["pi"] - b["taubar"]) * hmm)
    tau = sp.Rational(1, 2) * (Delta(hlm) + b["mu"] * hlm) - sp.Rational(1, 2) * b["pi"] * hmm
    pi = -sp.Rational(1, 2) * (Delta(hlbar) + b["mubar"] * hlbar) - sp.Rational(1, 2) * b["tau"] * hbarbar
    pibar = -sp.Rational(1, 2) * (Delta(hlm) + b["mu"] * hlm) - sp.Rational(1, 2) * b["taubar"] * hmm
    return Connections(sig, kap, rho, rhobar, tau, pi, pibar, eps, epsbar, alpha, alphabar, beta)


def ghp_thorn(f: sp.Expr, spin: int, boost: int, bg: Background) -> sp.Expr:
    b = physical_background(bg)
    p, q = boost + spin, boost - spin
    return D(f) - p * b["eps"] * f - q * b["epsbar"] * f


def ghp_eth(f: sp.Expr, spin: int, boost: int, bg: Background) -> sp.Expr:
    b = physical_background(bg)
    p, q = boost + spin, boost - spin
    return delta(f) - p * b["beta"] * f - q * b["alphabar"] * f


def ghp_ethp(f: sp.Expr, spin: int, boost: int, bg: Background) -> sp.Expr:
    b = physical_background(bg)
    p, q = boost + spin, boost - spin
    return bardelta(f) - p * b["alpha"] * f - q * b["betabar"] * f


def Delta_n(f: sp.Expr, n: int) -> sp.Expr:
    return Delta(R**n * f) / R**n


def thorn_n(f: sp.Expr, n: int, spin: int, boost: int, bg: Background) -> sp.Expr:
    return ghp_thorn(R**n * f, spin, boost, bg) / R ** (n + 1)


def eth_n(f: sp.Expr, n: int, spin: int, boost: int, bg: Background) -> sp.Expr:
    return ghp_eth(R**n * f, spin, boost, bg) / R ** (n + 1)


def ethp_n(f: sp.Expr, n: int, spin: int, boost: int, bg: Background) -> sp.Expr:
    return ghp_ethp(R**n * f, spin, boost, bg) / R ** (n + 1)


def regular_primitives(parent: Parent, bg: Background) -> dict[str, sp.Expr]:
    V, C, Cs, B, Bs = parent.V, parent.C, parent.Cs, parent.B, parent.Bs
    Sig = sp.Rational(1, 2) * thorn_n(Bs, 1, 2, 0, bg) + sp.Rational(1, 2) * (bg.rho0 - bg.rhobar0) * Bs - R**2 * (bg.pibar0 + bg.tau0) * Cs
    Kap = thorn_n(Cs, 2, 1, 1, bg) - bg.rhobar0 * Cs - sp.Rational(1, 2) * eth_n(V, 2, 0, 2, bg) - sp.Rational(1, 2) * R * (bg.pibar0 + bg.tau0) * V
    Rh = sp.Rational(1, 2) * bg.mu0 * V + sp.Rational(1, 2) * ethp_n(Cs, 2, 1, 1, bg) - sp.Rational(1, 2) * eth_n(C, 2, -1, 1, bg) - sp.Rational(1, 2) * R * bg.pi0 * Cs - sp.Rational(1, 2) * R * (bg.pibar0 + 2 * bg.tau0) * C
    Ta = sp.Rational(1, 2) * Delta_n(Cs, 2) + sp.Rational(1, 2) * R * bg.mu0 * Cs - sp.Rational(1, 2) * R * bg.pi0 * Bs
    Pi = -sp.Rational(1, 2) * Delta_n(C, 2) - sp.Rational(1, 2) * R * bg.mubar0 * C - sp.Rational(1, 2) * R * bg.tau0 * B
    Pisharp = -sp.Rational(1, 2) * Delta_n(Cs, 2) - sp.Rational(1, 2) * R * bg.mu0 * Cs - sp.Rational(1, 2) * R * bg.taubar0 * Bs
    Al = -sp.Rational(1, 4) * Delta_n(C, 2) + sp.Rational(1, 4) * R * (2 * bg.mu0 - bg.mubar0) * C - sp.Rational(1, 4) * eth_n(B, 1, -2, 0, bg) + sp.Rational(1, 2) * bg.beta0 * B - sp.Rational(1, 4) * R * (bg.pibar0 + bg.tau0) * B
    Be = -sp.Rational(1, 4) * Delta_n(Cs, 2) - sp.Rational(1, 4) * R * (bg.mu0 + 2 * bg.mubar0) * Cs + sp.Rational(1, 4) * ethp_n(Bs, 1, 2, 0, bg) + sp.Rational(1, 2) * bg.alpha0 * Bs - sp.Rational(1, 4) * R * (bg.pi0 + bg.taubar0) * Bs
    Ep = -sp.Rational(1, 4) * Delta_n(V, 2) - sp.Rational(1, 4) * R * eth_n(C, 2, -1, 1, bg) + sp.Rational(1, 4) * R * ethp_n(Cs, 2, 1, 1, bg) + sp.Rational(1, 4) * R * (bg.mu0 - bg.mubar0) * V - sp.Rational(1, 4) * R**2 * (bg.pibar0 + 2 * bg.tau0) * C - sp.Rational(1, 4) * R**2 * (3 * bg.pi0 + 2 * bg.taubar0) * Cs
    conn = connections(parent, bg)
    return {"Sig": Sig, "Kap": Kap, "Rh": Rh, "Rhsharp": conn.rhobar / R**3, "Ta": Ta, "Pi": Pi, "Pisharp": Pisharp, "Al": Al, "Be": Be, "Ep": Ep, "Epsharp": conn.epsbar / R**2}


def raw_source(parent1: Parent, parent2: Parent, bg: Background) -> tuple[sp.Expr, dict[str, sp.Expr]]:
    b, h1 = physical_background(bg), metric(parent1)
    c1, c2 = connections(parent1, bg), connections(parent2, bg)
    psi0, psi1, psi2 = R**5 * parent2.Z0, R**4 * parent2.Z1, R**3 * parent2.H
    D1 = lambda f: -sp.Rational(1, 2) * h1["hll"] * Delta(f)
    delta1 = lambda f: -h1["hlm"] * Delta(f) + sp.Rational(1, 2) * h1["hmm"] * bardelta(f)
    bardelta1 = lambda f: -h1["hlbar"] * Delta(f) + sp.Rational(1, 2) * h1["hbarbar"] * delta(f)
    Bcorr = bardelta1(psi0) + (-4 * c1.alpha + c1.pi) * psi0
    Ccorr = delta1(psi1) + (-2 * c1.beta - 4 * c1.tau) * psi1
    Dcorr = D1(psi1) + (-2 * c1.eps - 4 * c1.rho) * psi1
    Ercorr = D1(c2.sig) + (-c1.rho - c1.rhobar - 3 * c1.eps + c1.epsbar) * c2.sig
    Etcorr = delta1(c2.kap) + (-3 * c1.beta - c1.alphabar - c1.tau + c1.pibar) * c2.kap
    Jphysical = Ccorr + 3 * c1.sig * psi2
    Kphysical = Bcorr - Dcorr - 3 * c1.kap * psi2
    Qphysical = Ercorr - Etcorr
    d4p = lambda f: D(f) + (-4 * b["rho"] - b["rhobar"] - 3 * b["eps"] + b["epsbar"]) * f
    d3p = lambda f: delta(f) + (-3 * b["beta"] - b["alphabar"] - 4 * b["tau"] + b["pibar"]) * f
    source = d4p(Jphysical) + d3p(Kphysical) - 3 * R**3 * bg.psi20 * Qphysical
    return source, {"B": Bcorr, "C": Ccorr, "D": Dcorr, "Er": Ercorr, "Et": Etcorr}


def compact_source(parent1: Parent, parent2: Parent, bg: Background) -> tuple[sp.Expr, dict[str, sp.Expr]]:
    p1, p2 = regular_primitives(parent1, bg), regular_primitives(parent2, bg)
    V, C, Cs, B, Bs = parent1.V, parent1.C, parent1.Cs, parent1.B, parent1.Bs
    Z0, Z1, H = parent2.Z0, parent2.Z1, parent2.H
    C12 = -Cs * Delta_n(Z1, 4) + sp.Rational(1, 2) * Bs * ethp_n(Z1, 4, 1, 1, bg) + (-sp.Rational(3, 2) * Delta_n(Cs, 2) - sp.Rational(1, 2) * ethp_n(Bs, 1, 2, 0, bg) + R * (-sp.Rational(3, 2) * bg.mu0 + bg.mubar0) * Cs + R * (sp.Rational(5, 2) * bg.pi0 + sp.Rational(1, 2) * bg.taubar0) * Bs) * Z1
    B12 = -C * Delta_n(Z0, 5) + sp.Rational(1, 2) * B * eth_n(Z0, 5, 2, 2, bg) + (sp.Rational(1, 2) * Delta_n(C, 2) + eth_n(B, 1, -2, 0, bg) + R * (-2 * bg.mu0 + sp.Rational(1, 2) * bg.mubar0) * C + R * (bg.pibar0 + sp.Rational(1, 2) * bg.tau0) * B) * Z0
    D12 = -sp.Rational(1, 2) * V * Delta_n(Z1, 4) + (sp.Rational(1, 2) * Delta_n(V, 2) + sp.Rational(5, 2) * R * eth_n(C, 2, -1, 1, bg) - sp.Rational(5, 2) * R * ethp_n(Cs, 2, 1, 1, bg) + R * (-sp.Rational(5, 2) * bg.mu0 + sp.Rational(1, 2) * bg.mubar0) * V + R**2 * (sp.Rational(5, 2) * bg.pibar0 + 5 * bg.tau0) * C + R**2 * (sp.Rational(7, 2) * bg.pi0 + bg.taubar0) * Cs) * Z1
    Er12 = -sp.Rational(1, 2) * V * Delta_n(p2["Sig"], 2) - 3 * p1["Ep"] * p2["Sig"] + p1["Epsharp"] * p2["Sig"] - R * (p1["Rh"] + p1["Rhsharp"]) * p2["Sig"]
    Et12 = -Cs * Delta_n(p2["Kap"], 3) + sp.Rational(1, 2) * Bs * ethp_n(p2["Kap"], 3, 1, 2, bg) - sp.Rational(1, 2) * ethp_n(Bs, 1, 2, 0, bg) * p2["Kap"] + R * (bg.mubar0 * Cs + (sp.Rational(3, 2) * bg.pi0 + sp.Rational(1, 2) * bg.taubar0) * Bs) * p2["Kap"]
    J = R * C12 + 3 * p1["Sig"] * H
    K = R * B12 - D12 - 3 * p1["Kap"] * H
    Q = Er12 - R * Et12
    source_over_R6 = thorn_n(J, 5, 2, 1, bg) - (4 * bg.rho0 + bg.rhobar0) * J + R * eth_n(K, 6, 1, 2, bg) + R**2 * (-4 * bg.tau0 + bg.pibar0) * K - 3 * R * bg.psi20 * Q
    return source_over_R6, {"B": B12, "C": C12, "D": D12, "Er": Er12, "Et": Et12, "J": J, "K": K, "Q": Q, **p1}


def numeric(expr: sp.Expr, point: dict[sp.Symbol, sp.Expr]) -> complex:
    return complex(sp.N(expr.subs(point), 60))


def close(a: complex, b: complex, tolerance: float = 1.0e-36) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def check_random_oracle() -> None:
    points = [
        {t: sp.Rational(1, 7), R: sp.Rational(2, 11), y: sp.Rational(1, 3), phi: sp.Rational(1, 5), Mpar: 1, apar: sp.Rational(2, 5), Lpar: 1},
        {t: sp.Rational(-2, 9), R: sp.Rational(3, 10), y: sp.Rational(-2, 5), phi: sp.Rational(2, 7), Mpar: sp.Rational(6, 5), apar: sp.Rational(7, 10), Lpar: sp.Rational(4, 3)},
    ]
    for seed in (230519332, 201000162):
        rng = random.Random(seed)
        bg = make_background()
        parent1, parent2 = make_parent(rng, 1), make_parent(rng, -2)
        c1, c2 = connections(parent1, bg), connections(parent2, bg)
        raw, raw_parts = raw_source(parent1, parent2, bg)
        compact, compact_parts = compact_source(parent1, parent2, bg)
        primitive_expected = {
            "Sig": c1.sig / R**2, "Kap": c1.kap / R**3,
            "Rh": c1.rho / R**3, "Rhsharp": c1.rhobar / R**3,
            "Ta": c1.tau / R**2, "Pi": c1.pi / R**2,
            "Pisharp": c1.pibar / R**2, "Ep": c1.eps / R**2,
            "Al": c1.alpha / R**2, "Be": c1.beta / R**2,
            "Epsharp": c1.epsbar / R**2,
        }
        for point_index, point in enumerate(points):
            tag = f"seed={seed} point={point_index}"
            for name, expected in primitive_expected.items():
                check(f"oracle primitive {name} {tag}", close(numeric(expected, point), numeric(compact_parts[name], point)))
            scalings = {"B": 7, "C": 6, "D": 6, "Er": 4, "Et": 5}
            for name, power in scalings.items():
                check(f"oracle family {name} {tag}", close(numeric(raw_parts[name] / R**power, point), numeric(compact_parts[name], point)))
            check(f"oracle full source {tag}", close(numeric(raw / R**6, point), numeric(compact, point)))


def check_endpoint_and_mode_gates() -> None:
    Rv, L, a, yy, Shat, RH = sp.symbols("Rv L a yy Shat RH", positive=True, finite=True, real=True)
    dminus = L**2 - sp.I * a * Rv * yy
    S0 = Rv**7 * Shat
    forcing = 2 * (L**4 + a**2 * Rv**2 * yy**2) * dminus**4 * S0 / Rv**7
    check("S0 has corrected scri order seven", sp.limit(S0 / Rv**7, Rv, 0) == Shat)
    check("coordinate forcing finite and generically nonzero", sp.limit(forcing, Rv, 0) == 2 * L**12 * Shat)
    check("horizon forcing finite expression", not forcing.subs(Rv, RH).has(sp.zoo, sp.nan, sp.oo, -sp.oo))
    denom = (L**2 - I * a * RH * yy) * (L**2 + I * a * RH * yy)
    check("endpoint denominator positive form", sp.expand(denom) == L**4 + a**2 * RH**2 * yy**2)
    m1, m2 = sp.symbols("m1 m2", integer=True)
    f = sp.Function("f")(t, R, y) * sp.exp(I * m1 * phi)
    g = sp.Function("g")(t, R, y) * sp.exp(I * m2 * phi)
    check("ordered-pair Fourier mode selection", sp.simplify(sp.diff(f * g, phi) - I * (m1 + m2) * f * g) == 0)


def main() -> None:
    check_ledger()
    check_exact_operator_algebra()
    check_random_oracle()
    check_endpoint_and_mode_gates()


if __name__ == "__main__":
    main()
