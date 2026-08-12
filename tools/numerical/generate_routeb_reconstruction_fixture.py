#!/usr/bin/env python3
"""Generate an independent high-precision Route-B reconstruction fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mpmath as mp


FIELD_COUNT = 7
MODE_COUNT = 2


def profile(kind: int, mode: int, radius):
    amplitude = (mp.mpf("0.17") + mp.mpf("0.031") * (kind + 1) +
                 mp.mpf("0.019") * mode +
                 1j * (-mp.mpf("0.11") + mp.mpf("0.013") * kind -
                       mp.mpf("0.017") * mode))
    alpha = mp.mpf("0.61") + mp.mpf("0.037") * kind + mp.mpf("0.023") * mode
    oscillatory = (mp.mpf("0.021") - mp.mpf("0.014") * 1j) * (kind + 1)
    beta = mp.mpf("1.43") + mp.mpf("0.029") * kind + mp.mpf("0.017") * mode
    return amplitude * mp.exp(alpha * radius) + oscillatory * mp.sin(beta * radius)


def background(radius):
    spin, length, theta = mp.mpf("0.63"), mp.mpf("1.4"), mp.mpf("0.82")
    length2 = length**2
    angular = length2 - 1j * spin * radius * mp.cos(theta)
    radial = -length2 + 1j * spin * radius * mp.cos(theta)
    real_pi = length2**2 + spin**2 * radius**2 * mp.cos(theta)**2
    mu = 1 / radial
    tau = 1j * spin * mp.sin(theta) / (mp.sqrt(2) * angular**2)
    pi = -1j * spin * mp.sin(theta) / (mp.sqrt(2) * real_pi)
    return mu, tau, pi


def time_solve(value, delta, falloff: int):
    length2, mass = mp.mpf("1.4")**2, mp.mpf("1")
    return lambda r: ((delta(r) - r**2 / length2 * mp.diff(value, r) -
                       falloff * r / length2 * value(r)) /
                      (2 + 4 * mass * r / length2))


def f_function(level: int, mode: int):
    return lambda r: profile(40 + 3 * level, mode, r)


def angular_function(level: int, mode: int, slot: int):
    return lambda r: profile(80 + 17 * level + slot, mode, r)


def advance_mode(current, sharp, level: int, mode: int):
    f = f_function(level, mode)
    eth1f = angular_function(level, mode, 0)
    eth2g = angular_function(level, mode, 1)
    eth2c = angular_function(level, mode, 2)
    eth2pi = angular_function(level, mode, 3)
    ethp1bs = angular_function(level, mode, 4)
    ethp2cs = angular_function(level, mode, 5)

    def delta_g(r):
        mu, tau, _ = background(r)
        return -4 * r * mu * current[0](r) + eth1f(r) - r * tau * f(r)

    def delta_lam(r):
        mu, _, _ = background(r)
        return -r * (mu + mp.conj(mu)) * current[1](r) - f(r)

    def delta_h(r):
        mu, tau, _ = background(r)
        return (-3 * r * mu * current[2](r) + eth2g(r) -
                2 * r * tau * current[0](r))

    def delta_b(r):
        mu, _, _ = background(r)
        return r * (mu - mp.conj(mu)) * current[3](r) - 2 * current[1](r)

    def delta_pi(r):
        mu, tau, pi = background(r)
        return (-current[0](r) - r * (mp.conj(pi) + tau) * current[1](r) +
                mp.mpf("0.5") * r**2 * mu * (mp.conj(pi) + tau) *
                current[3](r))

    def delta_c(r):
        mu, tau, _ = background(r)
        return (-r * mp.conj(mu) * current[5](r) - 2 * current[4](r) -
                r * tau * current[3](r))

    def sharp_value(field: int, r):
        return mp.conj(sharp[field](r))

    def delta_u(r):
        mu, tau, pi = background(r)
        return (-r * mp.conj(mu) * current[6](r) - r * mu * eth2c(r) -
                r**2 * mu * (mp.conj(pi) + 2 * tau) * current[5](r) -
                2 * eth2pi(r) - 2 * r * mp.conj(pi) * current[4](r) -
                2 * current[2](r) - 2 * r * pi * sharp_value(4, r) -
                r * pi * ethp1bs(r) + r**2 * pi**2 * sharp_value(3, r) +
                r * mu * ethp2cs(r) +
                r**2 * (-3 * mu * pi + 2 * mp.conj(mu) * pi -
                        2 * mu * mp.conj(tau)) * sharp_value(5, r))

    return [
        time_solve(current[0], delta_g, 2),
        time_solve(current[1], delta_lam, 1),
        time_solve(current[2], delta_h, 3),
        time_solve(current[3], delta_b, 1),
        time_solve(current[4], delta_pi, 2),
        time_solve(current[5], delta_c, 2),
        time_solve(current[6], delta_u, 3),
    ]


def advance(states, level: int):
    return [advance_mode(states[mode], states[1 - mode], level, mode)
            for mode in range(MODE_COUNT)]


def format_complex(value) -> str:
    return ("teuk::Complex(" + format(float(mp.re(value)), ".17e") + ", " +
            format(float(mp.im(value)), ".17e") + ")")


def render() -> str:
    mp.mp.dps = 100
    states = [[lambda r, field=field, mode=mode: profile(field, mode, r)
               for field in range(FIELD_COUNT)] for mode in range(MODE_COUNT)]
    tower = [states]
    for level in range(4):
        states = advance(states, level)
        tower.append(states)
    horizon = mp.mpf("1.4")**2 / (1 + mp.sqrt(1 - mp.mpf("0.63")**2))
    values = [[[[tower[level][mode][field](radius)
                 for field in range(FIELD_COUNT)]
                for radius in (mp.mpf("0"), horizon)]
               for mode in range(MODE_COUNT)] for level in range(5)]
    if abs(values[1][0][0][0] - values[1][1][0][0]) < mp.mpf("1e-30"):
        raise RuntimeError("signed-mode h1 fixture values are not distinct")
    canonical = "\n".join(
        f"{level},{mode},{endpoint},{field}:" +
        mp.nstr(values[level][mode][endpoint][field], 100)
        for level in range(5) for mode in range(MODE_COUNT)
        for endpoint in range(2) for field in range(FIELD_COUNT)
    ) + "\n"
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    background_point = mp.mpf("0.317")
    background_coefficients = [
        [mp.diff(lambda x, slot=slot: background(x)[slot], background_point,
                 order) / mp.factorial(order)
         for order in range(5)]
        for slot in range(3)
    ]
    lines = [
        "#pragma once", "", "#include <array>",
        "#include \"teuk/types.hpp\"", "",
        "namespace routeb_reconstruction_fixture {", "",
        f'inline constexpr const char* sha256 = "{digest}";',
        "inline const std::array<std::array<teuk::Complex, 5>, 3>",
        "    background_coefficients{{",
    ]
    for slot in range(3):
        lines.append("  {{" + ", ".join(
            format_complex(background_coefficients[slot][order])
            for order in range(5)) + "}},")
    lines.extend([
        "}};", "",
        "inline const std::array<std::array<std::array<std::array<teuk::Complex, 7>, 2>, 2>, 5>",
        "    endpoint_values{{",
    ])
    for level in range(5):
        lines.append("  {{")
        for mode in range(MODE_COUNT):
            lines.append("    {{")
            for endpoint in range(2):
                lines.append("      {{" + ", ".join(
                    format_complex(values[level][mode][endpoint][field])
                    for field in range(FIELD_COUNT)) + "}},")
            lines.append("    }},")
        lines.append("  }}" + ("," if level < 4 else ""))
    lines.extend(["}};", "", "}  // namespace routeb_reconstruction_fixture", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if args.output.read_text() != expected:
            raise SystemExit(f"stale Route-B reconstruction fixture: {args.output}")
        print("Route-B reconstruction high-precision fixture freshness: PASS")
    else:
        args.output.write_text(expected)


if __name__ == "__main__":
    main()
