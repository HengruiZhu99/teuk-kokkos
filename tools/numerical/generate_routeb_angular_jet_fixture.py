#!/usr/bin/env python3
"""Generate an independent modal/radial oracle for the closed Route-B graph.

The oracle deliberately does not import or call production code.  It evaluates
spin-weighted harmonics from a factorial Wigner-d sum, constructs
Gauss-Legendre quadrature with mpmath, applies modal raise/lower/projection,
and propagates normalized radial Taylor coefficients with its own list algebra.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mpmath as mp


# Set precision before constructing any mp value.  mpmath constants retain the
# precision at construction and cannot recover digits after a later dps raise.
mp.mp.dps = 90

MODES = (-2, 2)
ELL_MAX = 5
NODE_COUNT = 13
MASS = mp.mpf("1")
SPIN = mp.mpf("0.63")
LENGTH = mp.mpf("1.4")
DAMPING = mp.mpf("0.31")
FIELD_SPINS = (-2, -2, -2, -1, -2, 0, -2, -1, -1, 0)


def add(a, b):
    n = min(len(a), len(b))
    return [a[k] + b[k] for k in range(n)]


def neg(a):
    return [-x for x in a]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    n = min(len(a), len(b))
    return [sum(a[j] * b[k - j] for j in range(k + 1)) for k in range(n)]


def scale(c, a):
    return [c * x for x in a]


def reciprocal(a):
    out = [1 / a[0]]
    for k in range(1, len(a)):
        out.append(-sum(a[j] * out[k - j] for j in range(1, k + 1)) / a[0])
    return out


def div(a, b):
    return mul(a, reciprocal(b))


def deriv(a):
    return [(k + 1) * a[k + 1] for k in range(len(a) - 1)]


def trunc(a, degree):
    return a[: degree + 1]


def conj(a):
    return [mp.conj(x) for x in a]


def const(value, degree):
    return [value] + [mp.mpc(0)] * degree


def radial(radius, degree):
    out = const(radius, degree)
    if degree:
        out[1] = 1
    return out


def factorial(n):
    return mp.factorial(n)


def wigner_d(ell, mp_, m, theta):
    k_min = max(0, m - mp_)
    k_max = min(ell + m, ell - mp_)
    norm = mp.sqrt(factorial(ell + m) * factorial(ell - m) *
                   factorial(ell + mp_) * factorial(ell - mp_))
    c, s = mp.cos(theta / 2), mp.sin(theta / 2)
    value = mp.mpf(0)
    for k in range(k_min, k_max + 1):
        denominator = (factorial(ell + m - k) * factorial(k) *
                       factorial(mp_ - m + k) * factorial(ell - mp_ - k))
        term = norm / denominator * c ** (2 * ell + m - mp_ - 2 * k) * \
            s ** (mp_ - m + 2 * k)
        if (mp_ - m + k) % 2:
            term = -term
        value += term
    return value


def swsh(ell, m, spin, theta):
    phase = -1 if abs(spin) % 2 else 1
    return phase * mp.sqrt((2 * ell + 1) / (4 * mp.pi)) * \
        wigner_d(ell, m, -spin, theta)


def angular_grid():
    x, weights = mp.gauss_quadrature(NODE_COUNT, "legendre")
    pairs = sorted((x[k], weights[k]) for k in range(NODE_COUNT))
    return [mp.acos(p[0]) for p in pairs], [p[1] for p in pairs]


THETA, WEIGHTS = [], []


def analyze(nodal, spin, m):
    ell_min = max(abs(spin), abs(m))
    return [sum(2 * mp.pi * WEIGHTS[n] * swsh(ell, m, spin, THETA[n]) *
                nodal[n] for n in range(NODE_COUNT))
            for ell in range(ell_min, ELL_MAX + 1)]


def synthesize(modal, spin, m):
    ell_min = max(abs(spin), abs(m))
    return [sum(swsh(ell, m, spin, THETA[n]) * modal[ell - ell_min]
                for ell in range(ell_min, ELL_MAX + 1))
            for n in range(NODE_COUNT)]


def project_jets(nodal_jets, spin, m):
    degree = len(nodal_jets[0]) - 1
    result = [[mp.mpc(0)] * (degree + 1) for _ in range(NODE_COUNT)]
    for order in range(degree + 1):
        values = [nodal_jets[n][order] for n in range(NODE_COUNT)]
        projected = synthesize(analyze(values, spin, m), spin, m)
        for n in range(NODE_COUNT):
            result[n][order] = projected[n]
    return result


def angular_jets(nodal_jets, spin, m, operation):
    degree = len(nodal_jets[0]) - 1
    target_spin = spin
    result = [[mp.mpc(0)] * (degree + 1) for _ in range(NODE_COUNT)]
    for order in range(degree + 1):
        modal = analyze([nodal_jets[n][order] for n in range(NODE_COUNT)], spin, m)
        ell_min = max(abs(spin), abs(m))
        if operation == "lap":
            operated = [-(ell - spin) * (ell + spin + 1) * modal[ell - ell_min]
                        for ell in range(ell_min, ELL_MAX + 1)]
        elif operation == "raise":
            target_spin = spin + 1
            operated = [mp.sqrt((ell - spin) * (ell + spin + 1)) *
                        modal[ell - ell_min] for ell in range(ell_min, ELL_MAX + 1)]
        else:
            target_spin = spin - 1
            operated = [-mp.sqrt((ell + spin) * (ell - spin + 1)) *
                        modal[ell - ell_min] for ell in range(ell_min, ELL_MAX + 1)]
        target_min = max(abs(target_spin), abs(m))
        aligned = [mp.mpc(0)] * (ELL_MAX - target_min + 1)
        for ell in range(max(ell_min, target_min), ELL_MAX + 1):
            aligned[ell - target_min] = operated[ell - ell_min]
        nodal = synthesize(aligned, target_spin, m)
        for n in range(NODE_COUNT):
            result[n][order] = nodal[n]
    return result


def profile_jet(kind, mode_index, ell, radius, degree):
    amplitude = (mp.mpf("0.11") + mp.mpf("0.013") * (kind + 1) +
                 mp.mpf("0.017") * mode_index + mp.mpf("0.009") * ell +
                 1j * (-mp.mpf("0.07") + mp.mpf("0.006") * kind -
                       mp.mpf("0.011") * mode_index))
    alpha = mp.mpf("0.47") + mp.mpf("0.021") * kind + mp.mpf("0.013") * ell
    osc = (mp.mpf("0.015") - mp.mpf("0.008") * 1j) * (kind + 1)
    beta = mp.mpf("1.31") + mp.mpf("0.019") * kind + mp.mpf("0.023") * ell
    return [amplitude * alpha**k * mp.exp(alpha * radius) / factorial(k) +
            osc * mp.diff(lambda r: mp.sin(beta * r), radius, k) / factorial(k)
            for k in range(degree + 1)]


def initial_field(kind, spin, mode_index, radius):
    m = MODES[mode_index]
    ell_min = max(abs(spin), abs(m))
    result = [[mp.mpc(0)] * 5 for _ in range(NODE_COUNT)]
    for ell in range(ell_min, min(ELL_MAX, ell_min + 2) + 1):
        jet = profile_jet(kind, mode_index, ell, radius, 4)
        for n in range(NODE_COUNT):
            y = swsh(ell, m, spin, THETA[n])
            result[n] = add(result[n], scale(y, jet))
    return result


def ghp(value, dt, spin, boost, m, radius0, prime):
    degree = len(value[0]) - 1
    pure = angular_jets(value, spin, m, "lower" if prime else "raise")
    output = []
    for n in range(NODE_COUNT):
        r = radial(radius0, degree)
        sign = 1j * SPIN * mp.cos(THETA[n]) if prime else -1j * SPIN * mp.cos(THETA[n])
        denominator = add(const(LENGTH**2, degree), scale(sign, r))
        dt_factor = 1j * SPIN * mp.sin(THETA[n]) if prime else -1j * SPIN * mp.sin(THETA[n])
        weight = -spin + boost if prime else spin + boost
        connection = (1j if prime else -1j) * weight * SPIN * mp.sin(THETA[n])
        first = div(add(scale(dt_factor, dt[n]), pure[n]), denominator)
        second = div(scale(connection, mul(r, value[n])), mul(denominator, denominator))
        output.append(scale(1 / mp.sqrt(2), add(first, second)))
    return output


def primary_coefficients(radius0, theta, m, degree):
    r = radial(radius0, degree)
    one = const(1, degree)
    r2 = mul(r, r)
    l2, l4, a2 = LENGTH**2, LENGTH**4, SPIN**2
    time = sub(mul(scale(8 * MASS, sub(scale(2 * MASS, one), scale(a2 / l2, r))),
                   add(one, scale(2 * MASS / l2, r))),
               scale(a2 * mp.sin(theta)**2, one))
    adv = add(sub(scale(l2, one), scale((8 * MASS**2 - a2) / l2, r2)),
              scale(4 * a2 * MASS / l4, mul(r2, r)))
    principal = scale(1 / l4, mul(r2, add(sub(scale(l4, one), scale(2 * l2 * MASS, r)),
                                           scale(a2, r2))))
    definition = add(scale(2j * SPIN * m, add(one, scale(4 * MASS / l2, r))),
                     add(scale(4 * MASS, add(scale(2, one), scale(-3 * a2 / l4, r2))),
                         add(scale(-2 * a2 / l2, r), scale(-4j * SPIN * mp.cos(theta), one))))
    q = add(scale(2, mul(r, add(add(scale(-1, one), scale(-MASS / l2, r)),
                                  scale(2 * a2 / l4, r2)))),
            scale(-2j * SPIN * m / l2, r2))
    psi = add(scale(-2, mul(r, add(scale(-MASS / l2, one), scale(-a2 / l4, r)))),
              scale(-2j * SPIN * m / l2, r))
    return time, adv, principal, definition, q, psi


def primary_step(state, lap, radius0, theta, m):
    degree = len(state[0]) - 1
    out_degree = degree - 1
    time, adv, principal, definition, qcoef, psicoef = primary_coefficients(radius0, theta, m, degree)
    velocity = div(sub(add(state[0], scale(2, mul(adv, state[1]))),
                       mul(definition, state[2])), time)
    p = add(add(mul(trunc(principal, out_degree), deriv(state[1])),
                mul(trunc(qcoef, out_degree), trunc(state[1], out_degree))),
            add(mul(trunc(psicoef, out_degree), trunc(state[2], out_degree)), lap))
    q = sub(deriv(velocity), scale(DAMPING,
            sub(trunc(state[1], out_degree), deriv(state[2]))))
    return p, q, trunc(velocity, out_degree)


def background(radius0, theta, degree):
    r = radial(radius0, degree)
    one = const(1, degree)
    c, s = mp.cos(theta), mp.sin(theta)
    mu = div(one, add(scale(-LENGTH**2, one), scale(1j * SPIN * c, r)))
    tau = div(scale(1j * SPIN * s / mp.sqrt(2), one),
              mul(sub(scale(LENGTH**2, one), scale(1j * SPIN * c, r)),
                  sub(scale(LENGTH**2, one), scale(1j * SPIN * c, r))))
    pi = div(scale(-1j * SPIN * s / mp.sqrt(2), one),
             add(scale(LENGTH**4, one), scale(SPIN**2 * c**2, mul(r, r))))
    return mu, tau, pi


def time_solve(value, delta, falloff, radius0):
    out_degree = len(delta) - 1
    r = radial(radius0, out_degree)
    radial_terms = add(scale(1 / LENGTH**2, mul(mul(r, r), deriv(value))),
                       scale(falloff / LENGTH**2, mul(r, trunc(value, out_degree))))
    denominator = add(const(2, out_degree), scale(4 * MASS / LENGTH**2, r))
    return div(sub(delta, radial_terms), denominator)


def reconstruction_step(states, primary_old, primary_new, radius0):
    degree = len(states[0][0][0]) - 1
    out_degree = degree - 1
    next_states = [[[[mp.mpc(0)] * (out_degree + 1) for _ in range(7)]
                    for _ in range(NODE_COUNT)] for _ in MODES]
    for mi, m in enumerate(MODES):
        eth_f = ghp([primary_old[mi][n][2] for n in range(NODE_COUNT)],
                    [primary_new[mi][n][2] for n in range(NODE_COUNT)],
                    -2, -2, m, radius0, False)
        for n in range(NODE_COUNT):
            mu, tau, _ = background(radius0, THETA[n], out_degree)
            r = radial(radius0, out_degree)
            g, lam = trunc(states[mi][n][0], out_degree), trunc(states[mi][n][1], out_degree)
            f = trunc(primary_old[mi][n][2], out_degree)
            dg = add(add(scale(-4, mul(mul(r, mu), g)), eth_f[n]), neg(mul(mul(r, tau), f)))
            dl = sub(neg(mul(mul(r, add(mu, conj(mu))), lam)), f)
            next_states[mi][n][0] = time_solve(states[mi][n][0], dg, 2, radius0)
            next_states[mi][n][1] = time_solve(states[mi][n][1], dl, 1, radius0)
    for mi, m in enumerate(MODES):
        projected_g = project_jets([next_states[mi][n][0] for n in range(NODE_COUNT)], -1, m)
        projected_l = project_jets([next_states[mi][n][1] for n in range(NODE_COUNT)], -2, m)
        for n in range(NODE_COUNT):
            next_states[mi][n][0], next_states[mi][n][1] = projected_g[n], projected_l[n]
        eth_g = ghp([states[mi][n][0] for n in range(NODE_COUNT)], projected_g,
                    -1, -1, m, radius0, False)
        for n in range(NODE_COUNT):
            mu, tau, pi0 = background(radius0, THETA[n], out_degree)
            r, r2 = radial(radius0, out_degree), mul(radial(radius0, out_degree), radial(radius0, out_degree))
            old = [trunc(states[mi][n][f], out_degree) for f in range(7)]
            dg, dl, h, b, piv, c, u = old
            dh = add(add(scale(-3, mul(mul(r, mu), h)), eth_g[n]), scale(-2, mul(mul(r, tau), dg)))
            db = sub(mul(mul(r, sub(mu, conj(mu))), b), scale(2, dl))
            dpi = add(add(neg(dg), neg(mul(mul(r, add(conj(pi0), tau)), dl))),
                      scale(mp.mpf("0.5"), mul(mul(mul(r2, mu), add(conj(pi0), tau)), b)))
            dc = add(add(neg(mul(mul(r, conj(mu)), c)), scale(-2, piv)), neg(mul(mul(r, tau), b)))
            next_states[mi][n][2] = time_solve(states[mi][n][2], dh, 3, radius0)
            next_states[mi][n][3] = time_solve(states[mi][n][3], db, 1, radius0)
            next_states[mi][n][4] = time_solve(states[mi][n][4], dpi, 2, radius0)
            next_states[mi][n][5] = time_solve(states[mi][n][5], dc, 2, radius0)
        for field, spin in ((2, 0), (3, -2), (4, -1), (5, -1)):
            projected = project_jets([next_states[mi][n][field] for n in range(NODE_COUNT)], spin, m)
            for n in range(NODE_COUNT):
                next_states[mi][n][field] = projected[n]
    for mi, m in enumerate(MODES):
        eth_c = ghp([states[mi][n][5] for n in range(NODE_COUNT)],
                    [next_states[mi][n][5] for n in range(NODE_COUNT)], -1, 1, m, radius0, False)
        eth_pi = ghp([states[mi][n][4] for n in range(NODE_COUNT)],
                     [next_states[mi][n][4] for n in range(NODE_COUNT)], -1, 0, m, radius0, False)
        partner = 1 - mi
        bs = [conj(states[partner][n][3]) for n in range(NODE_COUNT)]
        dtbs = [conj(next_states[partner][n][3]) for n in range(NODE_COUNT)]
        cs = [conj(states[partner][n][5]) for n in range(NODE_COUNT)]
        dtcs = [conj(next_states[partner][n][5]) for n in range(NODE_COUNT)]
        ethp_b = ghp(bs, dtbs, 2, 0, m, radius0, True)
        ethp_c = ghp(cs, dtcs, 1, 1, m, radius0, True)
        for n in range(NODE_COUNT):
            mu, tau, pi0 = background(radius0, THETA[n], out_degree)
            r, r2 = radial(radius0, out_degree), mul(radial(radius0, out_degree), radial(radius0, out_degree))
            old = [trunc(states[mi][n][f], out_degree) for f in range(7)]
            g, lam, h, b, piv, c, u = old
            pi_sharp, b_sharp, c_sharp = trunc(conj(states[partner][n][4]), out_degree), trunc(bs[n], out_degree), trunc(cs[n], out_degree)
            du = add(neg(mul(mul(r, conj(mu)), u)), neg(mul(mul(r, mu), eth_c[n])))
            du = add(du, neg(mul(mul(mul(r2, mu), add(conj(pi0), scale(2, tau))), c)))
            du = add(du, scale(-2, eth_pi[n]))
            du = add(du, scale(-2, mul(mul(r, conj(pi0)), piv)))
            du = add(du, scale(-2, h))
            du = add(du, scale(-2, mul(mul(r, pi0), pi_sharp)))
            du = add(du, neg(mul(mul(r, pi0), ethp_b[n])))
            du = add(du, mul(mul(mul(r2, pi0), pi0), b_sharp))
            du = add(du, mul(mul(r, mu), ethp_c[n]))
            coefficient = add(add(scale(-3, mul(mu, pi0)), scale(2, mul(conj(mu), pi0))),
                              scale(-2, mul(mu, conj(tau))))
            du = add(du, mul(mul(r2, coefficient), c_sharp))
            next_states[mi][n][6] = time_solve(states[mi][n][6], du, 3, radius0)
        projected_u = project_jets([next_states[mi][n][6] for n in range(NODE_COUNT)], 0, m)
        for n in range(NODE_COUNT):
            next_states[mi][n][6] = projected_u[n]
    return next_states


def endpoint_tower(radius0):
    primary = [[[initial_field(field, -2, mi, radius0)[n] for field in range(3)]
                for n in range(NODE_COUNT)] for mi in range(2)]
    reconstruction = [[[initial_field(3 + field, FIELD_SPINS[3 + field], mi, radius0)[n]
                        for field in range(7)] for n in range(NODE_COUNT)] for mi in range(2)]
    for mi, m in enumerate(MODES):
        for field in range(3):
            projected = project_jets([primary[mi][n][field] for n in range(NODE_COUNT)], -2, m)
            for n in range(NODE_COUNT): primary[mi][n][field] = projected[n]
        for field in range(7):
            spin = FIELD_SPINS[3 + field]
            projected = project_jets([reconstruction[mi][n][field] for n in range(NODE_COUNT)], spin, m)
            for n in range(NODE_COUNT): reconstruction[mi][n][field] = projected[n]
    tower = [(primary, reconstruction)]
    for _level in range(4):
        degree = len(primary[0][0][0]) - 1
        next_primary = [[[None] * 3 for _ in range(NODE_COUNT)] for _ in MODES]
        for mi, m in enumerate(MODES):
            lap = angular_jets([primary[mi][n][2] for n in range(NODE_COUNT)], -2, m, "lap")
            for n in range(NODE_COUNT):
                state = tuple(primary[mi][n][f] for f in range(3))
                result = primary_step(state, lap[n], radius0, THETA[n], m)
                for field in range(3): next_primary[mi][n][field] = result[field]
            for field in range(3):
                projected = project_jets([next_primary[mi][n][field] for n in range(NODE_COUNT)], -2, m)
                for n in range(NODE_COUNT): next_primary[mi][n][field] = projected[n]
        next_reconstruction = reconstruction_step(reconstruction, primary, next_primary, radius0)
        primary, reconstruction = next_primary, next_reconstruction
        assert len(primary[0][0][0]) == degree
        tower.append((primary, reconstruction))
    return tower


def format_complex(value):
    return ("teuk::Complex(" + format(float(mp.re(value)), ".17e") + ", " +
            format(float(mp.im(value)), ".17e") + ")")


def render():
    global THETA, WEIGHTS
    # Prove the scientific path has not silently collapsed to binary64.
    if SPIN == mp.mpf(float(SPIN)) or LENGTH == mp.mpf(float(LENGTH)) or \
            DAMPING == mp.mpf(float(DAMPING)):
        raise RuntimeError("Route-B angular oracle constants were float-rounded")
    THETA, WEIGHTS = angular_grid()
    horizon = LENGTH**2 / (MASS + mp.sqrt(MASS**2 - SPIN**2))
    sample_radii = (mp.mpf(0), horizon / 2, horizon)
    towers = [endpoint_tower(radius) for radius in sample_radii]
    values = [[[[[None for _n in range(NODE_COUNT)] for _field in range(10)]
                 for _mode in MODES] for _sample in range(3)] for _level in range(5)]
    for sample in range(3):
        for level in range(5):
            primary, reconstruction = towers[sample][level]
            for mi in range(2):
                for n in range(NODE_COUNT):
                    for field in range(3): values[level][sample][mi][field][n] = primary[mi][n][field][0]
                    for field in range(7): values[level][sample][mi][3 + field][n] = reconstruction[mi][n][field][0]
    if abs(values[4][0][0][0][3] - values[4][0][1][0][3]) < mp.mpf("1e-30"):
        raise RuntimeError("signed h4 oracle values are not distinct")
    canonical = "\n".join(
        f"{l},{e},{m},{f},{n}:" + mp.nstr(values[l][e][m][f][n], 80)
        for l in range(5) for e in range(3) for m in range(2)
        for f in range(10) for n in range(NODE_COUNT)) + "\n"
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    lines = ["#pragma once", "", "#include <array>", '#include "teuk/types.hpp"', "",
             "namespace routeb_angular_fixture {", "",
             f'inline constexpr const char* sha256 = "{digest}";',
             f"inline constexpr int node_count = {NODE_COUNT};",
             "inline const std::array<std::array<std::array<std::array<std::array<teuk::Complex, 13>, 10>, 2>, 3>, 5>",
             "    sample_values{{"]
    for l in range(5):
        lines.append("  {{")
        for e in range(3):
            lines.append("    {{")
            for m in range(2):
                lines.append("      {{")
                for f in range(10):
                    lines.append("        {{" + ", ".join(format_complex(values[l][e][m][f][n]) for n in range(NODE_COUNT)) + "}},")
                lines.append("      }},")
            lines.append("    }},")
        lines.append("  }}" + ("," if l < 4 else ""))
    lines.extend(["}};", "", "}  // namespace routeb_angular_fixture", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if args.output.read_text() != expected:
            raise SystemExit(f"stale Route-B angular fixture: {args.output}")
        print("Route-B angular modal/radial fixture freshness: PASS")
    else:
        args.output.write_text(expected)


if __name__ == "__main__":
    main()
