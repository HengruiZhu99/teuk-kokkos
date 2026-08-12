#!/usr/bin/env python3
"""Generate an independent coordinate-Weyl fixture for Route-B curvature.

The fixture starts from analytic signed-mode ORG metric components.  It builds
the coordinate metric directly from the Ripley code tetrad and varies the full
Weyl tensor, including Ricci trace subtraction and the prescribed perturbed
tetrad contribution to Psi1.  It never calls the C++ connection, NP curvature,
endpoint-extraction, or Route-B provider algebra under test.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

SYMBOLIC = Path(__file__).resolve().parents[1] / "symbolic"
sys.path.insert(0, str(SYMBOLIC))

from verify_linear_psi0_kerr_ad_oracle import (  # noqa: E402
    DIMENSION,
    I,
    PHI_INDEX,
    R_INDEX,
    THETA_INDEX,
    T_INDEX,
    Jet2,
    coordinate_linearized_curvature,
    jet_cos,
    jet_exp,
    jet_sin,
    kerr_geometry,
    reconstruct_complex_org_metric,
)


MASS = 1.0
LENGTH = 2.0
TIME = 0.19
PHI = 0.27
ELL = 4
PROVIDER_ELL_MAX = 12
THETA_COUNT = 32
RADIAL_COUNT = 33
RADIAL_MAX = 0.8
SPINS = (0.63, 0.999, -0.74)
MODES = (-2, 2)
RADIAL_INDICES = (8, 16, 24)
THETA_INDICES = (6, 16, 25)
LEVELS = (0, 1, 2)
CASES = (0, 1, 2, 3)  # V only, C only, B only, all fields
FIELD_SPINS = (0, -1, -2)  # V, C, B

# Mode order is (-2,+2).  V is conjugate-paired because h_ll is one real
# tetrad component; C and B are independent signed coefficients and their
# sharp values are selected from the opposite mode.
AMPLITUDES = (
    (0.31 + 0.17j, 0.31 - 0.17j),
    (-0.21 + 0.13j, 0.27 - 0.09j),
    (0.18 - 0.16j, -0.14 + 0.23j),
)
LAMBDAS = (
    (0.23 + 0.11j, 0.23 - 0.11j),
    (-0.17 + 0.08j, 0.19 - 0.14j),
    (0.13 - 0.12j, -0.09 + 0.16j),
)
ALPHAS = (0.37, 0.43, 0.51)
LINEAR = (0.14, -0.11, 0.09)
QUADRATIC = (0.06, 0.08, -0.05)


def wigner_d(ell: int, m_prime: int, m: int, theta: Jet2) -> Jet2:
    k_min = max(0, m - m_prime)
    k_max = min(ell + m, ell - m_prime)
    normalization = math.sqrt(
        math.factorial(ell + m)
        * math.factorial(ell - m)
        * math.factorial(ell + m_prime)
        * math.factorial(ell - m_prime)
    )
    cosine = jet_cos(0.5 * theta)
    sine = jet_sin(0.5 * theta)
    result = Jet2.constant(0.0)
    for k in range(k_min, k_max + 1):
        denominator = (
            math.factorial(ell + m - k)
            * math.factorial(k)
            * math.factorial(m_prime - m + k)
            * math.factorial(ell - m_prime - k)
        )
        term = (
            normalization
            / denominator
            * cosine ** (2 * ell + m - m_prime - 2 * k)
            * sine ** (m_prime - m + 2 * k)
        )
        if (m_prime - m + k) & 1:
            term = -term
        result = result + term
    return result


def swsh(mode: int, spin: int, theta: Jet2) -> Jet2:
    phase = -1.0 if abs(spin) & 1 else 1.0
    return (
        phase
        * math.sqrt((2 * ELL + 1) / (4 * math.pi))
        * wigner_d(ELL, mode, -spin, theta)
    )


def stored_coefficient(
    coordinates: tuple[Jet2, Jet2, Jet2, Jet2],
    field: int,
    mode_index: int,
    level: int,
) -> Jet2:
    time, radius, theta, _phi = coordinates
    mode = MODES[mode_index]
    radial = (
        radius**2
        * jet_exp(ALPHAS[field] * radius)
        * (1.0 + LINEAR[field] * radius + QUADRATIC[field] * radius**2)
    )
    base = (
        AMPLITUDES[field][mode_index]
        * jet_exp(LAMBDAS[field][mode_index] * time)
        * radial
        * swsh(mode, FIELD_SPINS[field], theta)
    )
    return LAMBDAS[field][mode_index] ** level * base


def expected_value(
    spin: float,
    radial_index: int,
    theta: float,
    mode_index: int,
    level: int,
    case_index: int,
) -> tuple[complex, complex]:
    radius_value = RADIAL_MAX * radial_index / (RADIAL_COUNT - 1)
    point = (TIME, radius_value, theta, PHI)
    coordinates = tuple(
        Jet2.variable(point[index], index) for index in range(DIMENSION)
    )
    _time, radius, _theta, phi = coordinates
    geometry = kerr_geometry(coordinates, MASS, spin, LENGTH)
    opposite = 1 - mode_index
    mode = MODES[mode_index]
    phase = jet_exp(I * mode * phi)
    active = lambda field: case_index == 3 or case_index == field
    zero = Jet2.constant(0.0)
    v = (stored_coefficient(coordinates, 0, mode_index, level)
         if active(0) else zero)
    c = (stored_coefficient(coordinates, 1, mode_index, level)
         if active(1) else zero)
    c_sharp = (stored_coefficient(
        coordinates, 1, opposite, level
    ).conjugate() if active(1) else zero)
    b = (stored_coefficient(coordinates, 2, mode_index, level)
         if active(2) else zero)
    b_sharp = (stored_coefficient(
        coordinates, 2, opposite, level
    ).conjugate() if active(2) else zero)
    h_ll = radius**2 * v * phase
    h_lm = radius**2 * c_sharp * phase
    h_lbar = radius**2 * c * phase
    h_mm = radius * b_sharp * phase
    h_barbar = radius * b * phase
    perturbation = reconstruct_complex_org_metric(
        geometry, h_ll, h_lm, h_lbar, h_mm, h_barbar
    )
    psi0, psi1, maximum_ricci, background_psi0, background_psi1, _ = (
        coordinate_linearized_curvature(geometry, perturbation, h_lm)
    )
    if maximum_ricci > 3.0e-11 or abs(background_psi0) > 2.0e-11 or \
            abs(background_psi1) > 2.0e-11:
        raise RuntimeError("coordinate-Weyl background validation failed")
    azimuth = cmath.exp(I * mode * PHI)
    return psi0 / azimuth / radius_value**5, psi1 / azimuth / radius_value**4


def format_complex(value: complex) -> str:
    return (
        "teuk::Complex(" + format(value.real, ".17e") + ", "
        + format(value.imag, ".17e") + ")"
    )


def render() -> str:
    nodes, _weights = np.polynomial.legendre.leggauss(THETA_COUNT)
    theta = [math.acos(float(node)) for node in nodes]
    entries = []
    canonical = []
    for case_index in CASES:
        for spin_index, spin in enumerate(SPINS):
            for radial_index in RADIAL_INDICES:
                for theta_index in THETA_INDICES:
                    for mode_index in range(len(MODES)):
                        for level in LEVELS:
                            z0, z1 = expected_value(
                                spin, radial_index, theta[theta_index],
                                mode_index, level, case_index
                            )
                            entries.append((case_index, spin_index,
                                            radial_index, theta_index,
                                            mode_index, level, z0, z1))
                            canonical.append(
                                f"{case_index},{spin_index},{radial_index},"
                                f"{theta_index},{mode_index},{level}:"
                                f"{z0.real:.17e},{z0.imag:.17e},"
                                f"{z1.real:.17e},{z1.imag:.17e}"
                            )
    digest = hashlib.sha256(("\n".join(canonical) + "\n").encode()).hexdigest()
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        '#include "teuk/types.hpp"',
        "",
        "namespace plus2_routeb_curvature_coordinate_fixture {",
        "",
        f'inline constexpr const char* sha256 = "{digest}";',
        f"inline constexpr double time = {TIME:.17e};",
        f"inline constexpr double phi = {PHI:.17e};",
        f"inline constexpr int ell = {ELL};",
        f"inline constexpr int provider_ell_max = {PROVIDER_ELL_MAX};",
        f"inline constexpr int theta_count = {THETA_COUNT};",
        f"inline constexpr int radial_count = {RADIAL_COUNT};",
        f"inline constexpr double radial_max = {RADIAL_MAX:.17e};",
        "inline constexpr std::array<int, 2> modes{{-2, 2}};",
        "inline constexpr std::array<double, 3> spins{{"
        + ", ".join(f"{value:.17e}" for value in SPINS) + "}};",
        "inline constexpr std::array<int, 3> field_spins{{0, -1, -2}};",
        "inline const std::array<std::array<teuk::Complex, 2>, 3> amplitudes{{",
    ]
    for values in AMPLITUDES:
        lines.append("  {{" + ", ".join(format_complex(v) for v in values) + "}},")
    lines.extend([
        "}};",
        "inline const std::array<std::array<teuk::Complex, 2>, 3> lambdas{{",
    ])
    for values in LAMBDAS:
        lines.append("  {{" + ", ".join(format_complex(v) for v in values) + "}},")
    lines.extend([
        "}};",
        "inline constexpr std::array<double, 3> alphas{{"
        + ", ".join(f"{v:.17e}" for v in ALPHAS) + "}};",
        "inline constexpr std::array<double, 3> linear{{"
        + ", ".join(f"{v:.17e}" for v in LINEAR) + "}};",
        "inline constexpr std::array<double, 3> quadratic{{"
        + ", ".join(f"{v:.17e}" for v in QUADRATIC) + "}};",
        "",
        "struct Expected {",
        "  int case_index;",
        "  int spin_index;",
        "  int radial_index;",
        "  int theta_index;",
        "  int mode_index;",
        "  int level;",
        "  teuk::Complex z0;",
        "  teuk::Complex z1;",
        "};",
        "",
        f"inline const std::array<Expected, {len(entries)}> expected{{{{",
    ])
    for entry in entries:
        ci, si, ri, ti, mi, level, z0, z1 = entry
        lines.append(
            f"  {{{ci}, {si}, {ri}, {ti}, {mi}, {level}, "
            f"{format_complex(z0)}, {format_complex(z1)}}},"
        )
    lines.extend([
        "}};",
        "",
        "}  // namespace plus2_routeb_curvature_coordinate_fixture",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if args.output.read_text() != expected:
            raise SystemExit(f"stale Route-B curvature coordinate fixture: {args.output}")
        print("Route-B curvature coordinate-Weyl fixture freshness: PASS")
    else:
        args.output.write_text(expected)


if __name__ == "__main__":
    main()
