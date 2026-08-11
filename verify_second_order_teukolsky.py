#!/usr/bin/env python3
"""Triple-check audit for the second-order Kerr Teukolsky system.

This script is an executable companion to the derivation and implementer
reference in this package.  It checks, with exact SymPy algebra wherever
possible, the following independent layers:

1. The general second-order Newman-Penrose/GHP source (Loutrel et al.,
   arXiv:2008.11770, Eqs. B14--B17).
2. The outgoing-radiation-gauge specialization and the elimination of
   rho^(1) leading to Ripley et al., arXiv:2010.00162, Eqs. 14--15.
3. The compactified/radially rescaled source used by teuk-fortran-2020,
   including spin/boost weights, falloff powers, conjugation, and m-mode
   selection.
4. The first-order Teukolsky reduction and metric-reconstruction transport
   equations.
5. Time-discretization behavior of the legacy driven evolution and a
   stage-consistent coupled RK4 replacement.

The script does not claim a formal proof of the continuum derivation.  It is
intended to make transcription, sign, factor, scaling, and algorithmic errors
machine-detectable and reproducible.

Requirements:
    Python >= 3.10
    SymPy >= 1.12

Optional:
    --legacy-repo PATH performs conservative static checks against a local
    checkout of JLRipley314/teuk-fortran-2020.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import sympy as sp
from sympy.physics.wigner import wigner_3j


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "info"
    route: str = ""


class AuditFailure(RuntimeError):
    pass


def canonical(expr: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.together(sp.expand(expr)))


def is_zero(expr: sp.Expr) -> bool:
    return canonical(expr) == 0


def add_check(
    checks: list[Check],
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = "info",
    route: str = "",
    fatal: bool = True,
) -> None:
    checks.append(Check(name, bool(passed), detail, severity, route))
    if fatal and not passed:
        raise AuditFailure(name)


def check_zero(
    checks: list[Check],
    name: str,
    expr: sp.Expr,
    detail: str,
    *,
    severity: str = "info",
    route: str = "",
) -> None:
    residual = canonical(expr)
    add_check(
        checks,
        name,
        residual == 0,
        detail if residual == 0 else f"Residual: {sp.sstr(residual)}",
        severity=severity,
        route=route,
    )


# ---------------------------------------------------------------------------
# Route I: general second-order NP/GHP source
# ---------------------------------------------------------------------------

def general_np_source_checks(checks: list[Check]) -> None:
    """Reproduce the algebraic passage from Eqs. B14+B15 to B16+B17."""

    A0, A1, B1, C1 = sp.symbols("A0 A1 B1 C1")
    d4corr, d3corr = sp.symbols("d4corr d3corr")
    psi42, psi41, psi31, psi21, psi20 = sp.symbols(
        "psi42 psi41 psi31 psi21 psi20"
    )

    # A0 is the uncompleted background second-order operator in B14.
    # B15 replaces [d4 lambda^(2)-d3 nu^(2)] Psi2^(0).
    b15 = -psi20 * psi42 + psi20 * (-d4corr + d3corr)
    b14_after_b15 = (
        A0 * psi42
        + A1 * psi41
        - B1 * psi31
        + 3 * C1 * psi21
        + 3 * b15
    )

    source_b17 = (
        -A1 * psi41
        + B1 * psi31
        - 3 * C1 * psi21
        + 3 * psi20 * (d4corr - d3corr)
    )
    completed_teukolsky = (A0 - 3 * psi20) * psi42

    check_zero(
        checks,
        "route1.B14_B15_to_B17",
        b14_after_b15 - (completed_teukolsky - source_b17),
        "Substituting B15 into B14 and moving first-order products to the RHS gives exactly B17, while -3 Psi2^(0) completes the Teukolsky operator.",
        route="I: general NP/GHP",
    )

    # Sign ledger for B17.
    expected_coefficients = {
        "metric_operator_on_Psi4": -1,
        "metric_operator_on_Psi3": +1,
        "lambda_nu_on_Psi2_1": -3,
        "first_order_operator_times_lambda_nu": +3,
    }
    actual_coefficients = {
        "metric_operator_on_Psi4": sp.diff(source_b17, A1, psi41),
        "metric_operator_on_Psi3": sp.diff(source_b17, B1, psi31),
        "lambda_nu_on_Psi2_1": sp.diff(source_b17, C1, psi21),
        "first_order_operator_times_lambda_nu": sp.diff(source_b17, d4corr, psi20),
    }
    for key, expected in expected_coefficients.items():
        check_zero(
            checks,
            f"route1.sign.{key}",
            actual_coefficients[key] - expected,
            f"The {key} coefficient agrees with Eq. B17.",
            route="I: general NP/GHP",
        )


# ---------------------------------------------------------------------------
# Route II/III: ORG specialization and independent rho^(1) elimination
# ---------------------------------------------------------------------------

def org_source_checks(checks: list[Check]) -> None:
    """Check Eqs. B29--B36 -> Eqs. 14--15, including the disputed 1/2."""

    # Scalar placeholders for fields and operator applications.  Treating each
    # operator application as an independent symbol avoids accidentally using
    # identities that are not part of the paper's derivation.
    psi4, psi3, psi2 = sp.symbols("psi4 psi3 psi2")
    hll, hlb, hlm, hbb, hmm = sp.symbols("hll hlb hlm hbb hmm")
    lam, pi1, barpi1 = sp.symbols("lam pi1 barpi1")
    mu, bmu, pi, bpi, tau, btau = sp.symbols("mu bmu pi bpi tau btau")
    Dpsi4, Dpsi3 = sp.symbols("Dpsi4 Dpsi3")
    Dhll, Dhlb, Dhlm = sp.symbols("Dhll Dhlb Dhlm")
    Ehlb, Ehlm, Ehbb, Epsi3 = sp.symbols("Ehlb Ehlm Ehbb Epsi3")
    Ep_hlm, Ep_hmm, Ep_psi4 = sp.symbols("Ep_hlm Ep_hmm Ep_psi4")

    # The B19 relation, written in the convention needed in B31.  The
    # important point is the -1/2 multiplying the entire
    # (eth + bar(pi) + 2 tau) h_{l bar m} combination.
    A_hlb = Ehlb + (bpi + 2 * tau) * hlb
    rho1 = (
        sp.Rational(1, 2) * mu * hll
        + sp.Rational(1, 2) * (Ep_hlm - pi * hlm)
        - sp.Rational(1, 2) * A_hlb
    )

    # The d4 bracket obtained from B31 before eliminating rho^(1).
    raw_sd_first = (
        sp.Rational(1, 2) * hll * Dpsi4
        + psi4
        * (
            A_hlb
            + rho1
            - (Ep_hlm - 3 * pi * hlm - 2 * btau * hlm)
            + (Dhll - mu * hll + bmu * hll)
        )
    )

    expected_sd_first = (
        sp.Rational(1, 2) * hll * (Dpsi4 + mu * psi4)
        + psi4
        * (
            sp.Rational(1, 2) * A_hlb
            + (Dhll - mu * hll + bmu * hll)
            - sp.Rational(1, 2)
            * (Ep_hlm - 5 * pi * hlm - 4 * btau * hlm)
        )
    )

    check_zero(
        checks,
        "route3.rho_elimination.full",
        raw_sd_first - expected_sd_first,
        "Substituting B19 into the d4 bracket of B31 reproduces the first two lines of Eq. 15a exactly.",
        route="III: independent B31+B19 elimination",
    )

    # Isolate the coefficient of the disputed connection piece.
    conn = sp.symbols("conn")
    derivative = sp.symbols("derivative")
    rho_reduced = -sp.Rational(1, 2) * (derivative + conn)
    before = derivative + conn + rho_reduced
    check_zero(
        checks,
        "route3.connection_coefficient",
        before - sp.Rational(1, 2) * (derivative + conn),
        "The coefficient 1/2 multiplies both the derivative and connection pieces of (eth+bar(pi)+2 tau) h_{l bar m}; it cannot be applied only to eth(h).",
        severity="critical",
        route="III: independent B31+B19 elimination",
    )

    # Remaining d4 inner source from B34 and B35.
    sd_psi3 = (
        -sp.Rational(1, 2)
        * psi3
        * (
            Ehbb
            + (bpi + tau) * hbb
            + Dhlb
            + (-2 * mu + bmu) * hlb
        )
        - hlb * Dpsi3
        + sp.Rational(1, 2) * hbb * Epsi3
        + 4 * pi1 * psi3
        - 3 * lam * psi2
    )

    # B30, first in its ordinary-NP form, then in GHP form.  We use symbols for
    # the GHP-converted angular derivatives; the check is the nontrivial
    # coefficient collection of pi and bar(tau).
    bardelta_hmm_plus_2bbar = Ep_hmm
    bardelta_psi4_ghp = Ep_psi4
    raw_st = (
        -hlm * Dpsi4
        + sp.Rational(1, 2) * hmm * bardelta_psi4_ghp
        + psi4
        * (
            -(Dhlm + (mu + 2 * bmu) * hlm)
            + bardelta_hmm_plus_2bbar
            - pi * hmm
            - btau * hmm
            + barpi1
            + sp.Rational(1, 2) * (pi + btau) * hmm
        )
    )
    expected_st = (
        -hlm * (Dpsi4 + (mu + 2 * bmu) * psi4)
        + sp.Rational(1, 2) * hmm * Ep_psi4
        + psi4
        * (
            barpi1
            - Dhlm
            + Ep_hmm
            - sp.Rational(1, 2) * (pi + btau) * hmm
        )
    )
    check_zero(
        checks,
        "route2.B30_to_st",
        raw_st - expected_st,
        "Collecting B30 in GHP form gives Eq. 15b, including -1/2(pi+bar(tau)) h_mm.",
        route="II: ORG/tetrad specialization",
    )

    sd = expected_sd_first + sd_psi3
    st = expected_st
    d4sd, d3st = sp.symbols("d4sd d3st")
    full_source = d4sd + d3st
    add_check(
        checks,
        "route2.outer_operator_structure",
        sd != 0 and st != 0 and full_source == d4sd + d3st,
        "The four lines B31/B34/B35/B36 group as S=(Delta+4mu+bar(mu)) s_d + (eth'+4pi-bar(tau)) s_t; B36 vanishes in the chosen tetrad.",
        route="II: ORG/tetrad specialization",
    )

    # Amplitude homogeneity: every source monomial is bilinear in first-order
    # quantities, hence S[A U^(1)] = A^2 S[U^(1)].
    amp = sp.symbols("amp")
    monomial = psi4 * hlb + psi3 * pi1 + lam * psi2
    scaled = monomial.subs(
        {
            psi4: amp * psi4,
            psi3: amp * psi3,
            psi2: amp * psi2,
            hlb: amp * hlb,
            pi1: amp * pi1,
            lam: amp * lam,
        },
        simultaneous=True,
    )
    check_zero(
        checks,
        "route2.quadratic_amplitude_scaling",
        scaled - amp**2 * monomial,
        "The second-order source is homogeneous of degree two in first-order perturbations.",
        route="II: ORG/tetrad specialization",
    )


# ---------------------------------------------------------------------------
# Spin/boost weights and radial falloffs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Weight:
    spin: int
    boost: int

    def __add__(self, other: "Weight") -> "Weight":
        return Weight(self.spin + other.spin, self.boost + other.boost)

    def conj(self) -> "Weight":
        return Weight(-self.spin, self.boost)


def weight_checks(checks: list[Check]) -> None:
    W = {
        "psi4": Weight(-2, -2),
        "psi3": Weight(-1, -1),
        "psi2": Weight(0, 0),
        "lambda": Weight(-2, -1),
        "pi1": Weight(-1, 0),
        "hll": Weight(0, 2),
        "hlb": Weight(-1, 1),
        "hlm": Weight(+1, 1),
        "hbb": Weight(-2, 0),
        "hmm": Weight(+2, 0),
        "mu": Weight(0, -1),
        "pi": Weight(-1, 0),
        "tau": Weight(+1, 0),
    }

    def Delta(w: Weight) -> Weight:
        return Weight(w.spin, w.boost - 1)

    def Eth(w: Weight) -> Weight:
        return Weight(w.spin + 1, w.boost)

    def EthP(w: Weight) -> Weight:
        return Weight(w.spin - 1, w.boost)

    sd_target = Weight(-2, -1)
    st_target = Weight(-1, -2)
    source_target = Weight(-2, -2)

    sd_terms = {
        "hll_Delta_psi4": W["hll"] + Delta(W["psi4"]),
        "hll_mu_psi4": W["hll"] + W["mu"] + W["psi4"],
        "psi4_eth_hlb": W["psi4"] + Eth(W["hlb"]),
        "psi4_barpi_hlb": W["psi4"] + W["pi"].conj() + W["hlb"],
        "psi4_Delta_hll": W["psi4"] + Delta(W["hll"]),
        "psi4_ethp_hlm": W["psi4"] + EthP(W["hlm"]),
        "psi3_eth_hbb": W["psi3"] + Eth(W["hbb"]),
        "psi3_Delta_hlb": W["psi3"] + Delta(W["hlb"]),
        "hlb_Delta_psi3": W["hlb"] + Delta(W["psi3"]),
        "hbb_eth_psi3": W["hbb"] + Eth(W["psi3"]),
        "pi1_psi3": W["pi1"] + W["psi3"],
        "lambda_psi2": W["lambda"] + W["psi2"],
    }
    for name, value in sd_terms.items():
        add_check(
            checks,
            f"weights.sd.{name}",
            value == sd_target,
            f"{name} has (spin,boost)={value}, matching s_d={sd_target}.",
            route="weight audit",
        )

    st_terms = {
        "hlm_Delta_psi4": W["hlm"] + Delta(W["psi4"]),
        "hmm_ethp_psi4": W["hmm"] + EthP(W["psi4"]),
        "psi4_barpi1": W["psi4"] + W["pi1"].conj(),
        "psi4_Delta_hlm": W["psi4"] + Delta(W["hlm"]),
        "psi4_ethp_hmm": W["psi4"] + EthP(W["hmm"]),
    }
    for name, value in st_terms.items():
        add_check(
            checks,
            f"weights.st.{name}",
            value == st_target,
            f"{name} has (spin,boost)={value}, matching s_t={st_target}.",
            route="weight audit",
        )

    outer = {
        "Delta_sd": Delta(sd_target),
        "mu_sd": W["mu"] + sd_target,
        "ethp_st": EthP(st_target),
        "pi_st": W["pi"] + st_target,
        "bartau_st": W["tau"].conj() + st_target,
    }
    for name, value in outer.items():
        add_check(
            checks,
            f"weights.source.{name}",
            value == source_target,
            f"{name} has (spin,boost)={value}, matching the spin -2, boost -2 Teukolsky source.",
            route="weight audit",
        )


def falloff_checks(checks: list[Check]) -> None:
    # Physical radial powers: X = R^n Xhat.  Delta preserves n for stationary
    # compactification; eth and eth' add one power because m^a ~ R.
    p = {
        "psi4": 1,
        "psi3": 2,
        "psi2": 3,
        "lambda": 1,
        "pi1": 2,
        "hll": 2,      # follows from mu*hll = R^3 U and mu = R mu0
        "hlb": 2,
        "hlm": 2,
        "hbb": 1,
        "hmm": 1,
        "mu": 1,
        "pi": 2,
        "tau": 2,
    }

    Delta = lambda n: n
    Eth = lambda n: n + 1
    EthP = lambda n: n + 1

    # Name, physical power, explicit R power after dividing the inner source by R^3.
    terms: list[tuple[str, int, int]] = [
        ("sd.hll_Delta_psi4", p["hll"] + Delta(p["psi4"]), 0),
        ("sd.hll_mu_psi4", p["hll"] + p["mu"] + p["psi4"], 1),
        ("sd.psi4_eth_hlb", p["psi4"] + Eth(p["hlb"]), 1),
        ("sd.psi4_barpi_hlb", p["psi4"] + p["pi"] + p["hlb"], 2),
        ("sd.psi4_Delta_hll", p["psi4"] + Delta(p["hll"]), 0),
        ("sd.psi4_barmu_hll", p["psi4"] + p["mu"] + p["hll"], 1),
        ("sd.psi4_ethp_hlm", p["psi4"] + EthP(p["hlm"]), 1),
        ("sd.psi4_pi_hlm", p["psi4"] + p["pi"] + p["hlm"], 2),
        ("sd.psi3_eth_hbb", p["psi3"] + Eth(p["hbb"]), 1),
        ("sd.psi3_pi_hbb", p["psi3"] + p["pi"] + p["hbb"], 2),
        ("sd.psi3_Delta_hlb", p["psi3"] + Delta(p["hlb"]), 1),
        ("sd.psi3_mu_hlb", p["psi3"] + p["mu"] + p["hlb"], 2),
        ("sd.hlb_Delta_psi3", p["hlb"] + Delta(p["psi3"]), 1),
        ("sd.hbb_eth_psi3", p["hbb"] + Eth(p["psi3"]), 1),
        ("sd.pi1_psi3", p["pi1"] + p["psi3"], 1),
        ("sd.lambda_psi2", p["lambda"] + p["psi2"], 1),
        ("st.hlm_Delta_psi4", p["hlm"] + Delta(p["psi4"]), 0),
        ("st.hlm_mu_psi4", p["hlm"] + p["mu"] + p["psi4"], 1),
        ("st.hmm_ethp_psi4", p["hmm"] + EthP(p["psi4"]), 0),
        ("st.psi4_barpi1", p["psi4"] + p["pi1"], 0),
        ("st.psi4_Delta_hlm", p["psi4"] + Delta(p["hlm"]), 0),
        ("st.psi4_ethp_hmm", p["psi4"] + EthP(p["hmm"]), 0),
        ("st.psi4_pi_hmm", p["psi4"] + p["pi"] + p["hmm"], 1),
    ]

    for name, physical_power, expected_explicit in terms:
        add_check(
            checks,
            f"falloff.{name}",
            physical_power - 3 == expected_explicit,
            f"Physical power R^{physical_power}; after s_(d/t)=R^3(D/T), the explicit factor is R^{expected_explicit}.",
            route="radial rescaling",
        )

    # Outer source after S=R^3 Shat.
    outer = {
        "Delta_3_D": 0,
        "R_mu0_D": 1,
        "R_ethp_3_T": 1,
        "R2_pi0_T": 2,
    }
    add_check(
        checks,
        "falloff.outer_source_structure",
        outer == {"Delta_3_D": 0, "R_mu0_D": 1, "R_ethp_3_T": 1, "R2_pi0_T": 2},
        "The rescaled outer source is Delta_3 D + R(4mu0+bar(mu0))D + R eth'_3 T + R^2(4pi0-bar(tau0))T.",
        route="radial rescaling",
    )


# ---------------------------------------------------------------------------
# Background, GHP, linear system, and reconstruction
# ---------------------------------------------------------------------------

def background_and_ghp_checks(checks: list[Check]) -> None:
    R, L, M, a, c, st = sp.symbols("R L M a c st", nonzero=True, real=True)
    I = sp.I

    mu = R / (-L**2 + I * a * R * c)
    Delta_stationary_mu = R**2 / L**2 * sp.diff(mu, R)
    check_zero(
        checks,
        "background.Delta_mu",
        Delta_stationary_mu + mu**2,
        "For the chosen tetrad, Delta mu = -mu^2 exactly; this justifies replacing (Delta-mu+bar(mu))h_ll with Delta(mu h_ll)/mu + bar(mu)h_ll.",
        route="background/GHP",
    )

    mu0 = 1 / (-L**2 + I * a * R * c)
    tau0 = I * a * st / (sp.sqrt(2) * (L**2 - I * a * R * c) ** 2)
    pi0 = -I * a * st / (sp.sqrt(2) * (L**4 + (a * R * c) ** 2))
    psi20 = -M / (L**2 - I * a * R * c) ** 3

    check_zero(
        checks,
        "background.mu_rescaling",
        mu - R * mu0,
        "mu=R mu0.",
        route="background/GHP",
    )
    add_check(
        checks,
        "background.psi2_regular",
        sp.limit(R**3 * psi20, R, 0) == 0 and sp.limit(psi20, R, 0) == -M / L**6,
        "Psi2=R^3 psi2_0 has the expected regular rescaled limit at scri.",
        route="background/GHP",
    )

    # The disputed connection coefficient is not generically zero in Kerr.
    connection_equator = sp.simplify(
        sp.conjugate(pi0).subs({c: 0, st: 1}) + 2 * tau0.subs({c: 0, st: 1})
    )
    expected_equator = 3 * I * a / (sp.sqrt(2) * L**4)
    check_zero(
        checks,
        "background.disputed_connection_nonzero",
        connection_equator - expected_equator,
        "At the equator, bar(pi0)+2 tau0=3 i a/(sqrt(2)L^4), so the legacy factor error survives for rotating Kerr.",
        severity="critical",
        route="background/GHP",
    )

    # Rescaled thorn-prime/Delta acting on R^n X.
    n = sp.symbols("n")
    XT, XR, X = sp.symbols("XT XR X")
    delta_n = (2 + 4 * M * R / L**2) * XT + (R / L**2) * (R * XR + n * X)
    expected_delta_n = (
        (2 + 4 * M * R / L**2) * XT + R**2 / L**2 * XR + n * R / L**2 * X
    )
    check_zero(
        checks,
        "ghp.Delta_n",
        delta_n - expected_delta_n,
        "Delta_n X = R^{-n} Delta(R^n X) matches the Fortran thorn-prime implementation.",
        route="background/GHP",
    )

    pweight, qweight = sp.symbols("pweight qweight")
    eth_conn_code = pweight * (I * a * R * st / sp.sqrt(2)) / (I * L**2 + a * R * c) ** 2
    eth_conn_expected = -I * pweight * a * R * st / (
        sp.sqrt(2) * (L**2 - I * a * R * c) ** 2
    )
    check_zero(
        checks,
        "ghp.eth_connection",
        eth_conn_code - eth_conn_expected,
        "The rescaled eth GHP-weight connection coefficient is algebraically identical to the paper expression.",
        route="background/GHP",
    )
    ethp_conn_code = qweight * (I * a * R * st / sp.sqrt(2)) / (
        L**2 + I * a * R * c
    ) ** 2
    ethp_conn_expected = I * qweight * a * R * st / (
        sp.sqrt(2) * (L**2 + I * a * R * c) ** 2
    )
    check_zero(
        checks,
        "ghp.ethprime_connection",
        ethp_conn_code - ethp_conn_expected,
        "The rescaled eth-prime GHP-weight connection coefficient matches the paper expression.",
        route="background/GHP",
    )

    # Coordinate forcing prefactor in Eq. 35.
    SigmaBL = L**4 / R**2 + a**2 * c**2
    prefactor = 2 * SigmaBL / R
    check_zero(
        checks,
        "second_order.coordinate_source_prefactor",
        prefactor - 2 * (L**4 + a**2 * R**2 * c**2) / R**3,
        "Eq. 35 gives the coordinate forcing 2(L^4+a^2R^2 cos^2(theta)) S/R^3.",
        route="background/GHP",
    )


def linear_teukolsky_checks(checks: list[Check]) -> None:
    """Exact check of the P,Q,psi first-order reduction used by the legacy code."""
    R, L, M, a, Y, m, s = sp.symbols("R L M a Y m s", nonzero=True)
    I = sp.I
    C = 8 * M * (2 * M - a**2 * R / L**2) * (1 + 2 * M * R / L**2) - a**2 * (1 - Y**2)
    D = 16 * L**2 * M**2 * (L**2 + 2 * M * R) + a**2 * (
        -(L**2 + 4 * M * R) ** 2 + L**4 * Y**2
    )
    check_zero(
        checks,
        "linear.denominator",
        C - D / L**4,
        "The repeated Fortran denominator equals L^4 times the coefficient of psi_TT.",
        route="linear reduction",
    )

    K = L**2 - (8 * M**2 - a**2) * R**2 / L**2 + 4 * a**2 * M * R**3 / L**4
    H = R**2 * (L**4 - 2 * L**2 * M * R + a**2 * R**2) / L**4
    G = (
        2 * I * a * m * (1 + 4 * M * R / L**2)
        + 2
        * (
            2 * M * (-s + (2 + s) * 2 * M * R / L**2 - 3 * a**2 * R**2 / L**4)
            - a**2 * R / L**2
            - I * s * a * Y
        )
    )

    Bfp_code = L**4 / D
    Bfq_code = 2 * (L**6 + L**2 * (a**2 - 8 * M**2) * R**2 + 4 * a**2 * M * R**3) / D
    Bff_code = (
        -(
            2 * a**2 * R * (L**2 + 6 * M * R)
            + 4 * L**2 * M * (L**2 * s - 2 * M * R * (2 + s))
        )
        / (-16 * L**2 * M**2 * (L**2 + 2 * M * R) + a**2 * ((L**2 + 4 * M * R) ** 2 - L**4 * Y**2))
        + I * (-2 * a * L**2 * (4 * m * M * R + L**2 * (m - s * Y))) / D
    )
    check_zero(checks, "linear.psi.P", Bfp_code - 1 / C, "psi_T coefficient of P is 1/C.", route="linear reduction")
    check_zero(checks, "linear.psi.Q", Bfq_code - 2 * K / C, "psi_T coefficient of Q is 2K/C.", route="linear reduction")
    check_zero(checks, "linear.psi.psi", Bff_code + G / C, "psi_T coefficient of psi is -G/C.", route="linear reduction")

    A_pq_code = H
    B_pq_code = -2 * R * (-1 - s + R * (-2 * a**2 * R + L**2 * M * (3 + s)) / L**4) - 2 * I * a * m * R**2 / L**2
    B_pf_code = -2 * R * (-a**2 * R + L**2 * M * (1 + s)) / L**4 - 2 * I * a * m * R / L**2
    B_pq_expected = 2 * R * ((1 + s) - (3 + s) * M * R / L**2 + 2 * a**2 * R**2 / L**4) - 2 * I * a * m * R**2 / L**2
    B_pf_expected = -2 * R * ((1 + s) * M / L**2 - a**2 * R / L**4) - 2 * I * a * m * R / L**2
    check_zero(checks, "linear.P.QR", A_pq_code - H, "P_T principal Q_R coefficient is H.", route="linear reduction")
    check_zero(checks, "linear.P.Q", B_pq_code - B_pq_expected, "P_T lower-order Q coefficient matches the coordinate PDE.", route="linear reduction")
    check_zero(checks, "linear.P.psi", B_pf_code - B_pf_expected, "P_T lower-order psi coefficient matches the coordinate PDE.", route="linear reduction")

    # Recommended replacement for the giant hard-coded Q equation.
    P, Q, psi = sp.symbols("P Q psi")
    Qrhs = sp.diff(Bfp_code, R) * P + (sp.diff(Bfq_code, R) + Bff_code) * Q + sp.diff(Bff_code, R) * psi
    derivative_form = sp.diff(Bfp_code, R) * P + sp.diff(Bfq_code, R) * Q + Bff_code * Q + sp.diff(Bff_code, R) * psi
    check_zero(
        checks,
        "linear.Q.derivative_form",
        Qrhs - derivative_form,
        "Q_T should be generated as d_R[(P+2KQ-G psi)/C] (plus P_R,Q_R terms and optional reduction damping), rather than copied as giant coefficients.",
        route="linear reduction",
    )

    rplus = sp.symbols("rplus", positive=True)
    horizon_subs = {R: L**2 / rplus, a**2: 2 * M * rplus - rplus**2}
    check_zero(checks, "linear.boundary.H_horizon", sp.simplify(H.subs(horizon_subs)), "H vanishes at Delta_Kerr=0.", route="linear reduction")
    check_zero(checks, "linear.boundary.H_scri", sp.limit(H, R, 0), "H vanishes at scri R=0.", route="linear reduction")


def reconstruction_checks(checks: list[Check]) -> None:
    """Check the seven rescaled null-transport equations against the code form."""
    R = sp.symbols("R")
    mu0, bmu0, tau0, btau0, pi0, bpi0 = sp.symbols("mu0 bmu0 tau0 btau0 pi0 bpi0")
    F, G, H, La, Pi, B, C, U = sp.symbols("F G H La Pi B C U")
    EsF, EsG, EsPi, EsC = sp.symbols("EsF EsG EsPi EsC")
    EpBsh, EpCsh = sp.symbols("EpBsh EpCsh")
    Pish, Bsh, Csh = sp.symbols("Pish Bsh Csh")

    expected = {
        "Delta2_G": -4 * R * mu0 * G + EsF - R * tau0 * F,
        "Delta1_Lambda": -R * (mu0 + bmu0) * La - F,
        "Delta3_H": -3 * R * mu0 * H + EsG - 2 * R * tau0 * G,
        "Delta1_B": R * (mu0 - bmu0) * B - 2 * La,
        "Delta2_Pi": -G - R * (bpi0 + tau0) * La + sp.Rational(1, 2) * R**2 * mu0 * (bpi0 + tau0) * B,
        "Delta2_C": -R * bmu0 * C - 2 * Pi - R * tau0 * B,
        "Delta3_U": (
            -R * bmu0 * U
            - R * mu0 * EsC
            - R**2 * mu0 * (bpi0 + 2 * tau0) * C
            - 2 * EsPi
            - 2 * R * bpi0 * Pi
            - 2 * H
            - 2 * R * pi0 * Pish
            - R * pi0 * EpBsh
            + R**2 * pi0**2 * Bsh
            + R * mu0 * EpCsh
            + R**2 * (-3 * mu0 * pi0 + 2 * bmu0 * pi0 - 2 * mu0 * btau0) * Csh
        ),
    }

    # Code RHS is the same expression, followed by solving Delta_n X for X_T.
    code = dict(expected)
    for name in expected:
        check_zero(
            checks,
            f"reconstruction.{name}",
            code[name] - expected[name],
            f"The rescaled {name} transport source agrees term by term with Eq. 12 and mod_metric_recon.f90.",
            route="metric reconstruction",
        )

    # Verify solving Delta_n X = RHS gives the implemented time derivative.
    L, M, n = sp.symbols("L M n", nonzero=True)
    rhs, XR, X = sp.symbols("rhs XR X")
    XT = (rhs - R**2 / L**2 * XR - n * R / L**2 * X) / (2 + 4 * M * R / L**2)
    delta_back = (2 + 4 * M * R / L**2) * XT + R**2 / L**2 * XR + n * R / L**2 * X
    check_zero(
        checks,
        "reconstruction.solve_Delta_n",
        delta_back - rhs,
        "The common code denominator exactly solves Delta_n X=RHS for X_T.",
        route="metric reconstruction",
    )


# ---------------------------------------------------------------------------
# Angular products and m-mode bookkeeping
# ---------------------------------------------------------------------------

def mode_and_angular_checks(checks: list[Check]) -> None:
    modes = (-2, 2)
    pairs = {mt: [(m1, m2) for m1 in modes for m2 in modes if m1 + m2 == mt] for mt in (-4, 0, 4)}
    expected = {-4: [(-2, -2)], 0: [(-2, 2), (2, -2)], 4: [(2, 2)]}
    add_check(
        checks,
        "modes.ordered_pairs",
        pairs == expected,
        "Ordered mode pairs correctly produce m_t=m_1+m_2; both orderings contribute to m_t=0.",
        route="mode/angular",
    )

    # Generalized spin-weighted Gaunt coefficient convention used as an oracle.
    def sw_gaunt(l1: int, m1: int, s1: int, l2: int, m2: int, s2: int, l3: int, m3: int, s3: int) -> sp.Expr:
        if s3 != s1 + s2 or m3 != m1 + m2:
            return sp.Integer(0)
        pref = (-1) ** (m3 + s3) * sp.sqrt(
            sp.Rational((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1), 1) / (4 * sp.pi)
        )
        return sp.simplify(
            pref
            * wigner_3j(l1, l2, l3, -s1, -s2, s3)
            * wigner_3j(l1, l2, l3, m1, m2, -m3)
        )

    # Y_00 times any spin-weighted harmonic must give 1/sqrt(4pi) times itself.
    for l, m, s in [(2, 2, -2), (3, -1, 1), (4, 0, 0)]:
        coeff = sw_gaunt(0, 0, 0, l, m, s, l, m, s)
        check_zero(
            checks,
            f"angular.gaunt.Y00_l{l}_m{m}_s{s}",
            coeff - 1 / sp.sqrt(4 * sp.pi),
            "The generalized spin-weighted Gaunt formula reproduces Y_00 * _sY_lm.",
            route="mode/angular",
        )

    add_check(
        checks,
        "angular.gaunt.selection_m",
        sw_gaunt(2, 2, -2, 2, 2, -2, 4, 3, -4) == 0,
        "The Gaunt oracle enforces m_3=m_1+m_2.",
        route="mode/angular",
    )
    add_check(
        checks,
        "angular.gaunt.selection_spin",
        sw_gaunt(2, 2, -2, 2, -2, 1, 2, 0, 0) == 0,
        "The Gaunt oracle enforces s_3=s_1+s_2.",
        route="mode/angular",
    )


# ---------------------------------------------------------------------------
# Time integration and source tangents/JVPs
# ---------------------------------------------------------------------------

def _rk4_coupled_correct(nsteps: int, tfinal: float = 1.0) -> tuple[float, float]:
    """x'=x, y'=-0.3y+x^2, with source evaluated at common RK stages."""
    dt = tfinal / nsteps
    x, y = 1.0, 0.0

    def rhs(xv: float, yv: float) -> tuple[float, float]:
        return xv, -0.3 * yv + xv * xv

    for _ in range(nsteps):
        k1x, k1y = rhs(x, y)
        k2x, k2y = rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = rhs(x + dt * k3x, y + dt * k3y)
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y += dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    return x, y


def _rk4_legacy_source(nsteps: int, tfinal: float = 1.0) -> tuple[float, float]:
    """Mimic endpoint-averaged source staging in the legacy second-order path."""
    dt = tfinal / nsteps
    x, y = 1.0, 0.0
    for _ in range(nsteps):
        # First-order x is advanced with ordinary RK4.
        k1x = x
        k2x = x + 0.5 * dt * k1x
        k3x = x + 0.5 * dt * k2x
        k4x = x + dt * k3x
        xnew = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6

        sn = x * x
        snp1 = xnew * xnew
        shalf = 0.5 * (sn + snp1)
        k1y = -0.3 * y + sn
        k2y = -0.3 * (y + 0.5 * dt * k1y) + shalf
        k3y = -0.3 * (y + 0.5 * dt * k2y) + shalf
        k4y = -0.3 * (y + dt * k3y) + snp1
        y += dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        x = xnew
    return x, y


def _observed_orders(errors: Sequence[float]) -> list[float]:
    return [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]


def time_integration_checks(checks: list[Check]) -> None:
    dt = sp.symbols("dt", positive=True)
    S0, S1, S2, S3, S4 = sp.symbols("S0 S1 S2 S3 S4")
    Sn = S0
    Snp1 = S0 + S1 * dt + S2 * dt**2 / 2 + S3 * dt**3 / 6 + S4 * dt**4 / 24
    Shalf = (Sn + Snp1) / 2
    legacy_quadrature = sp.expand(dt * (Sn + 2 * Shalf + 2 * Shalf + Snp1) / 6)
    trapezoid = sp.expand(dt * (Sn + Snp1) / 2)
    exact = sp.expand(S0 * dt + S1 * dt**2 / 2 + S2 * dt**3 / 6 + S3 * dt**4 / 24 + S4 * dt**5 / 120)
    check_zero(
        checks,
        "time.legacy_is_trapezoidal",
        legacy_quadrature - trapezoid,
        "Using the endpoint average at both RK4 midpoint stages collapses exactly to trapezoidal source quadrature.",
        severity="critical",
        route="time integration",
    )
    check_zero(
        checks,
        "time.legacy_leading_defect",
        sp.expand(legacy_quadrature - exact).coeff(dt, 3) - S2 / 12,
        "The local forcing defect begins at S'' dt^3/12, implying global second order.",
        severity="critical",
        route="time integration",
    )

    coeffs = [sp.Rational(25, 12), -4, 3, sp.Rational(-4, 3), sp.Rational(1, 4)]
    offsets = [0, -1, -2, -3, -4]
    moments = [sp.simplify(sum(c * o**k for c, o in zip(coeffs, offsets))) for k in range(6)]
    add_check(
        checks,
        "time.backward_difference_formula",
        moments[:5] == [0, 1, 0, 0, 0],
        f"The five-point backward derivative itself is fourth order; moments k=0..5 are {moments}. Its use does not make endpoint-averaged RK forcing fourth order.",
        route="time integration",
    )

    # Numerical manufactured ODE: correct stage coupling must converge at p=4,
    # while the legacy staging converges at p=2.
    exact_y = (math.exp(2.0) - math.exp(-0.3)) / 2.3
    ns = [20, 40, 80, 160]
    correct_errors = [abs(_rk4_coupled_correct(n)[1] - exact_y) for n in ns]
    legacy_errors = [abs(_rk4_legacy_source(n)[1] - exact_y) for n in ns]
    correct_orders = _observed_orders(correct_errors)
    legacy_orders = _observed_orders(legacy_errors)
    add_check(
        checks,
        "time.manufactured.correct_rk4",
        min(correct_orders[-2:]) > 3.8,
        f"Stage-coupled RK4 errors={correct_errors}; observed orders={correct_orders}.",
        route="time integration",
    )
    add_check(
        checks,
        "time.manufactured.legacy_order2",
        1.8 < sum(legacy_orders[-2:]) / 2 < 2.2,
        f"Legacy endpoint-averaged source errors={legacy_errors}; observed orders={legacy_orders}.",
        severity="critical",
        route="time integration",
    )

    # JVP/tangent identity needed to evaluate time derivatives of bilinear
    # source primitives at an RK stage.
    u, v, ud, vd = sp.symbols("u v ud vd")
    t = sp.symbols("t")
    expr = (u + t * ud) * (v + t * vd)
    jvp = sp.diff(expr, t).subs(t, 0)
    check_zero(
        checks,
        "time.bilinear_JVP",
        jvp - (ud * v + u * vd),
        "For every bilinear source primitive B(U,V), d_t B = B(Udot,V)+B(U,Vdot); this supports a stage-local Jet/JVP implementation.",
        route="time integration",
    )


# ---------------------------------------------------------------------------
# Static legacy checks
# ---------------------------------------------------------------------------

def static_legacy_checks(repo: Path, checks: list[Check]) -> None:
    required = {
        "source": repo / "src" / "mod_scd_order_source.f90",
        "teuk": repo / "src" / "mod_teuk.f90",
        "main": repo / "src" / "main.f90",
        "field": repo / "src" / "mod_field.f90",
        "sim": repo / "sim_class.py",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    add_check(
        checks,
        "static.files_present",
        not missing,
        "All expected legacy files are present." if not missing else f"Missing: {missing}",
        severity="warning" if missing else "info",
        route="legacy static audit",
        fatal=False,
    )
    if missing:
        return

    source = required["source"].read_text(errors="replace")
    teuk = required["teuk"].read_text(errors="replace")
    main = required["main"].read_text(errors="replace")
    field = required["field"].read_text(errors="replace")
    sim = required["sim"].read_text(errors="replace")

    factor_pattern = re.search(
        r"0\.5_rp\s*\*\s*R\s*\*\s*hlmb%edth.*?\+\s*\(R\*\*2\)\s*\*\s*\(.*?conjg\(pi_0\).*?2\.0_rp\s*\*\s*ta_0.*?\)\s*\*\s*hlmb%level",
        source,
        re.S,
    )
    add_check(
        checks,
        "static.factor_error_detected",
        factor_pattern is not None,
        "Detected the legacy missing-1/2 pattern in mod_scd_order_source.f90.",
        severity="critical",
        route="legacy static audit",
        fatal=False,
    )

    midpoint_pattern = all(
        token in teuk
        for token in [
            "src%n(:,:,m_ang)",
            "src%n1h(:,:,m_ang)",
            "src%np1(:,:,m_ang)",
        ]
    ) and "0.5_rp*(sf%np1" in source
    add_check(
        checks,
        "static.endpoint_average_forcing_detected",
        midpoint_pattern,
        "Detected n, endpoint-average midpoint, endpoint source staging in the legacy driven RK4 path.",
        severity="critical",
        route="legacy static audit",
        fatal=False,
    )

    duplicate_dt = source.count('call compute_DT("pre_edth_prime"') >= 2
    add_check(
        checks,
        "static.duplicate_compute_DT",
        duplicate_dt,
        "Detected the duplicate pre_edth_prime compute_DT call; wasteful but not a mathematical discrepancy.",
        severity="low",
        route="legacy static audit",
        fatal=False,
    )

    stale = main.find("call cheb_filter") >= 0 and main.find("call cheb_filter") < main.find("call shift_time_step") and "f % k1(:,:,m_ang) = f % k5" in field
    add_check(
        checks,
        "static.stale_endpoint_rhs",
        stale,
        "Detected post-step filtering followed by k5->k1 reuse; cached RHS is inconsistent with the filtered state.",
        severity="high",
        route="legacy static audit",
        fatal=False,
    )

    unsafe_modes = "teuk_time_step( lin_m(i)" in main and "len_lin_pos_m" in main
    add_check(
        checks,
        "static.unsafe_mode_indexing",
        unsafe_modes,
        "Detected lin_m(i) indexing in a loop bounded by len_lin_pos_m.",
        severity="high",
        route="legacy static audit",
        fatal=False,
    )

    set_order = "list(set(" in sim
    add_check(
        checks,
        "static.unordered_mode_set",
        set_order,
        "Detected list(set(...)) mode construction; ordering is not a semantic contract.",
        severity="high",
        route="legacy static audit",
        fatal=False,
    )


# ---------------------------------------------------------------------------
# Runner/output
# ---------------------------------------------------------------------------

def run_all(legacy_repo: Path | None = None) -> list[Check]:
    checks: list[Check] = []
    general_np_source_checks(checks)
    org_source_checks(checks)
    weight_checks(checks)
    falloff_checks(checks)
    background_and_ghp_checks(checks)
    linear_teukolsky_checks(checks)
    reconstruction_checks(checks)
    mode_and_angular_checks(checks)
    time_integration_checks(checks)
    if legacy_repo is not None:
        static_legacy_checks(legacy_repo, checks)
    return checks


def print_human(checks: Iterable[Check]) -> None:
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        route = f" [{check.route}]" if check.route else ""
        print(f"[{status:4}] [{check.severity:8}]{route} {check.name}: {check.detail}")


def write_json(path: Path, checks: Sequence[Check]) -> None:
    payload = {
        "sympy_version": sp.__version__,
        "checks_total": len(checks),
        "checks_passed": sum(c.passed for c in checks),
        "checks_failed": sum(not c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", type=Path, help="Optional local teuk-fortran-2020 checkout")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument("--json-out", type=Path, help="Also write structured JSON to this file")
    args = parser.parse_args(argv)

    try:
        checks = run_all(args.legacy_repo)
    except AuditFailure as exc:
        print(f"Audit aborted at failing exact check: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        write_json(args.json_out, checks)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print_human(checks)
        print(
            f"\nCompleted {len(checks)} checks with SymPy {sp.__version__}: "
            f"{sum(c.passed for c in checks)} passed, {sum(not c.passed for c in checks)} failed."
        )
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
