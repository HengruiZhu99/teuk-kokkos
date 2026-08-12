#!/usr/bin/env python3
"""Generate an independent high-precision Route-B primary-jet fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mpmath as mp


def coefficients(radius: mp.mpf):
    mass, spin, length = mp.mpf("1"), mp.mpf("0.63"), mp.mpf("1.4")
    theta, spin_weight, azimuthal_mode = mp.mpf("0.82"), -2, -2
    length2, length4 = length**2, length**4
    spin2 = spin**2
    time = (8 * mass * (2 * mass - spin2 * radius / length2) *
            (1 + 2 * mass * radius / length2) - spin2 * mp.sin(theta)**2)
    radial_advection = (length2 - (8 * mass**2 - spin2) * radius**2 / length2 +
                        4 * spin2 * mass * radius**3 / length4)
    radial_principal = (radius**2 / length4 *
                        (length4 - 2 * length2 * mass * radius +
                         spin2 * radius**2))
    definition = (2j * spin * azimuthal_mode *
                  (1 + 4 * mass * radius / length2) +
                  2 * (2 * mass *
                       (-spin_weight + (2 + spin_weight) *
                        2 * mass * radius / length2 -
                        3 * spin2 * radius**2 / length4) -
                       spin2 * radius / length2 +
                       1j * spin_weight * spin * mp.cos(theta)))
    q = (2 * radius *
         (1 + spin_weight - (3 + spin_weight) * mass * radius / length2 +
          2 * spin2 * radius**2 / length4) -
         2j * spin * azimuthal_mode * radius**2 / length2)
    psi = (-2 * radius *
           ((1 + spin_weight) * mass / length2 -
            spin2 * radius / length4) -
           2j * spin * azimuthal_mode * radius / length2)
    return time, radial_advection, radial_principal, definition, q, psi


def initial_functions():
    return [
        lambda r: (mp.mpf("0.47") - mp.mpf("0.18") * 1j) *
                  mp.exp(mp.mpf("1.31") * r) +
                  (mp.mpf("0.09") + mp.mpf("0.04") * 1j) *
                  mp.sin(mp.mpf("2.17") * r),
        lambda r: (mp.mpf("-0.36") + mp.mpf("0.27") * 1j) *
                  mp.exp(mp.mpf("1.73") * r) +
                  (mp.mpf("0.07") - mp.mpf("0.05") * 1j) *
                  mp.cos(mp.mpf("2.43") * r),
        lambda r: (mp.mpf("0.62") + mp.mpf("0.11") * 1j) *
                  mp.exp(mp.mpf("1.11") * r) +
                  (mp.mpf("-0.08") + mp.mpf("0.06") * 1j) *
                  mp.sin(mp.mpf("2.71") * r),
    ]


def angular_function(level: int):
    amplitude = (mp.mpf("0.13") * (level + 1) +
                 mp.mpf("0.04") * (level + 2) * 1j)
    alpha = mp.mpf("1.19") + mp.mpf("0.23") * level
    beta = mp.mpf("2.29") + mp.mpf("0.17") * level
    return lambda r: amplitude * mp.exp(alpha * r) + \
        (mp.mpf("0.03") - mp.mpf("0.02") * 1j) * (level + 1) * mp.sin(beta * r)


def advance(state, level: int):
    p, q_state, psi_state = state
    angular = angular_function(level)
    damping = mp.mpf("0.31")

    def velocity(r):
        time, radial_advection, _, definition, _, _ = coefficients(r)
        return (p(r) + 2 * radial_advection * q_state(r) -
                definition * psi_state(r)) / time

    def next_p(r):
        _, _, radial_principal, _, q_coefficient, psi_coefficient = coefficients(r)
        return (radial_principal * mp.diff(q_state, r) +
                q_coefficient * q_state(r) +
                psi_coefficient * psi_state(r) + angular(r))

    def next_q(r):
        return (mp.diff(velocity, r) -
                damping * (q_state(r) - mp.diff(psi_state, r)))

    return next_p, next_q, velocity


def format_complex(value) -> str:
    return ("teuk::Complex(" + format(float(mp.re(value)), ".17e") + ", " +
            format(float(mp.im(value)), ".17e") + ")")


def render() -> str:
    mp.mp.dps = 100
    length = mp.mpf("1.4")
    horizon = length**2 / (mp.mpf("1") + mp.sqrt(1 - mp.mpf("0.63")**2))
    states = [initial_functions()]
    for level in range(4):
        states.append(advance(states[-1], level))
    values = [[[states[level][field](radius) for field in range(3)]
               for radius in (mp.mpf("0"), horizon)]
              for level in range(5)]
    canonical = "\n".join(
        f"{level},{endpoint},{field}:" + mp.nstr(values[level][endpoint][field], 100)
        for level in range(5) for endpoint in range(2) for field in range(3)
    ) + "\n"
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    lines = [
        "#pragma once", "", "#include <array>",
        "#include \"teuk/types.hpp\"", "", "namespace routeb_primary_fixture {", "",
        f'inline constexpr const char* sha256 = "{digest}";',
        "inline const std::array<std::array<std::array<teuk::Complex, 3>, 2>, 5>",
        "    endpoint_values{{",
    ]
    for level in range(5):
        lines.append("  {{")
        for endpoint in range(2):
            lines.append("    {{" + ", ".join(
                format_complex(values[level][endpoint][field])
                for field in range(3)) + "}},")
        lines.append("  }}" + ("," if level < 4 else ""))
    lines.extend(["}};", "", "}  // namespace routeb_primary_fixture", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if args.output.read_text() != expected:
            raise SystemExit(f"stale Route-B Teukolsky fixture: {args.output}")
        print("Route-B Teukolsky high-precision fixture freshness: PASS")
    else:
        args.output.write_text(expected)


if __name__ == "__main__":
    main()
