#!/usr/bin/env python3
"""Generate an independent coordinate-Weyl fixture for Route-B curvature.

The fixture starts from analytic signed-mode ORG metric components.  It builds
the coordinate metric directly from the Ripley code tetrad and varies the full
Weyl tensor, including Ricci trace subtraction and the prescribed perturbed
tetrad contribution to Psi1.  It never calls the C++ connection, NP curvature,
endpoint-extraction, or Route-B provider algebra under test.

For the derivative adapter it applies the defining physical GHP operators to
the coordinate-Weyl contractions and differentiates those contractions at two
independent fourth-order spacings.  For scri it extrapolates the already
normalized coordinate-Weyl scalars, not the provider's curvature numerators.
"""

from __future__ import annotations

import argparse
import cmath
import functools
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
PROVIDER_ELL_MAX = 24
THETA_COUNT = 32
RADIAL_COUNT = 33
SPINS = (0.63, 0.999, -0.74)
MODES = (-2, 2)
RADIAL_INDICES = (8, 16, 32)
THETA_INDICES = (6, 16, 25)
LEVELS = (0, 1, 2)
CASES = (0, 1, 2, 3)  # V only, C only, B only, all fields
DERIVATIVE_CASES = (3,)  # full signed ORG perturbation
SCRI_CASES = (3,)  # complete signed ORG perturbation
FIELD_SPINS = (0, -1, -2)  # V, C, B
DERIVATIVE_STEP = 5.0e-4
SCRI_EXTRAPOLATION_WEIGHTS = (6.0, -15.0, 20.0, -15.0, 6.0, -1.0)

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


def future_horizon_radius(spin: float) -> float:
    r_plus = MASS + math.sqrt(MASS * MASS - spin * spin)
    return LENGTH * LENGTH / r_plus


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
    radial_power: int = 2,
) -> Jet2:
    time, radius, theta, _phi = coordinates
    mode = MODES[mode_index]
    radial = (
        radius**radial_power
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


@functools.cache
def coordinate_modal_curvature(
    spin: float,
    radius_value: float,
    theta: float,
    mode_index: int,
    level: int,
    case_index: int,
    radial_power: int = 2,
) -> tuple[complex, complex]:
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
    v = (
        stored_coefficient(coordinates, 0, mode_index, level, radial_power)
        if active(0)
        else zero
    )
    c = (
        stored_coefficient(coordinates, 1, mode_index, level, radial_power)
        if active(1)
        else zero
    )
    c_sharp = (
        stored_coefficient(coordinates, 1, opposite, level, radial_power)
        .conjugate()
        if active(1)
        else zero
    )
    b = (
        stored_coefficient(coordinates, 2, mode_index, level, radial_power)
        if active(2)
        else zero
    )
    b_sharp = (
        stored_coefficient(coordinates, 2, opposite, level, radial_power)
        .conjugate()
        if active(2)
        else zero
    )
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
    if (
        maximum_ricci > 3.0e-11
        or abs(background_psi0) > 2.0e-11
        or abs(background_psi1) > 2.0e-11
    ):
        raise RuntimeError("coordinate-Weyl background validation failed")
    azimuth = cmath.exp(I * mode * PHI)
    return psi0 / azimuth, psi1 / azimuth


def expected_value(
    spin: float,
    radial_index: int,
    theta: float,
    mode_index: int,
    level: int,
    case_index: int,
) -> tuple[complex, complex]:
    radius_value = (
        future_horizon_radius(spin) * radial_index / (RADIAL_COUNT - 1)
    )
    psi0, psi1 = coordinate_modal_curvature(
        spin, radius_value, theta, mode_index, level, case_index
    )
    return psi0 / radius_value**5, psi1 / radius_value**4


def fourth_order_first_derivative(
    function,
    coordinate: float,
    spacing: float,
    backward: bool,
) -> complex:
    if backward:
        return (
            25.0 * function(coordinate)
            - 48.0 * function(coordinate - spacing)
            + 36.0 * function(coordinate - 2.0 * spacing)
            - 16.0 * function(coordinate - 3.0 * spacing)
            + 3.0 * function(coordinate - 4.0 * spacing)
        ) / (12.0 * spacing)
    return (
        -function(coordinate + 2.0 * spacing)
        + 8.0 * function(coordinate + spacing)
        - 8.0 * function(coordinate - spacing)
        + function(coordinate - 2.0 * spacing)
    ) / (12.0 * spacing)


def richardson_first_derivative(
    function,
    coordinate: float,
    spacing: float,
    backward: bool = False,
) -> tuple[complex, float]:
    coarse = fourth_order_first_derivative(
        function, coordinate, spacing, backward
    )
    fine = fourth_order_first_derivative(
        function, coordinate, 0.5 * spacing, backward
    )
    extrapolated = fine + (fine - coarse) / 15.0
    return extrapolated, abs(fine - coarse) / 15.0


def expected_derivatives(
    spin: float,
    radial_index: int,
    theta: float,
    mode_index: int,
    case_index: int,
) -> tuple[tuple[complex, ...], tuple[complex, ...], float]:
    """Apply physical GHP operators directly to coordinate-Weyl scalars.

    This route never calls the compact provider's radial, modal, or point GHP
    implementations.  It differentiates the full coordinate-Weyl contractions
    and then applies the defining physical operators

      thorn' X = Delta X,
      eth X = (delta-p beta-q alphabar) X,
      eth' X = (bardelta-p alpha-q betabar) X,

    before removing the explicit powers of R.  gamma=0 and q=0 for Psi0 and
    Psi1 in the repository convention.
    """
    radius_max = future_horizon_radius(spin)
    radius = radius_max * radial_index / (RADIAL_COUNT - 1)
    mode = MODES[mode_index]
    coordinates = tuple(
        Jet2.variable(value, index)
        for index, value in enumerate((TIME, radius, theta, PHI))
    )
    geometry = kerr_geometry(coordinates, MASS, spin, LENGTH)
    n_vector = tuple(value.value for value in geometry.n)
    m_vector = tuple(value.value for value in geometry.m)
    mbar_vector = tuple(value.value for value in geometry.mbar)
    alpha = geometry.alpha.value
    beta = geometry.beta.value
    radial_spacing = DERIVATIVE_STEP * max(1.0, radius)
    theta_spacing = DERIVATIVE_STEP
    backward = radial_index == RADIAL_COUNT - 1
    values = [0.0j] * 8
    wrong_values = [0.0j] * 8
    maximum_remainder = 0.0

    for curvature_index, falloff, p_weight, angular_slot, delta_slot in (
        (1, 4, 2, 2, 0),  # Psi1 -> ethprime_4 and Delta_4
        (0, 5, 4, 6, 4),  # Psi0 -> eth_5 and Delta_5
    ):
        for tangent_level in (0, 1):
            raw, raw_t = (
                coordinate_modal_curvature(
                    spin, radius, theta, mode_index, level, case_index
                )[curvature_index]
                for level in (tangent_level, tangent_level + 1)
            )

            def radial_profile(radial_coordinate: float) -> complex:
                return coordinate_modal_curvature(
                    spin,
                    radial_coordinate,
                    theta,
                    mode_index,
                    tangent_level,
                    case_index,
                )[curvature_index]

            def angular_profile(theta_coordinate: float) -> complex:
                return coordinate_modal_curvature(
                    spin,
                    radius,
                    theta_coordinate,
                    mode_index,
                    tangent_level,
                    case_index,
                )[curvature_index]

            raw_r, radial_remainder = richardson_first_derivative(
                radial_profile, radius, radial_spacing, backward
            )
            raw_theta, theta_remainder = richardson_first_derivative(
                angular_profile, theta, theta_spacing
            )
            raw_phi = I * mode * raw
            directional_delta = sum(
                vector * derivative
                for vector, derivative in zip(
                    n_vector, (raw_t, raw_r, raw_theta, raw_phi), strict=True
                )
            )
            if curvature_index == 0:
                directional_angular = sum(
                    vector * derivative
                    for vector, derivative in zip(
                        m_vector,
                        (raw_t, raw_r, raw_theta, raw_phi),
                        strict=True,
                    )
                ) - p_weight * beta * raw
            else:
                directional_angular = sum(
                    vector * derivative
                    for vector, derivative in zip(
                        mbar_vector,
                        (raw_t, raw_r, raw_theta, raw_phi),
                        strict=True,
                    )
                ) - p_weight * alpha * raw
            values[delta_slot + tangent_level] = (
                directional_delta / radius**falloff
            )
            values[angular_slot + tangent_level] = (
                directional_angular / radius ** (falloff + 1)
            )
            normalized = raw / radius**falloff
            wrong_values[delta_slot + tangent_level] = (
                values[delta_slot + tangent_level]
                - falloff * n_vector[R_INDEX] * normalized / radius
            )
            wrong_values[angular_slot + tangent_level] = (
                sum(
                    vector * derivative
                    for vector, derivative in zip(
                        m_vector if curvature_index == 0 else mbar_vector,
                        (raw_t, raw_r, raw_theta, raw_phi),
                        strict=True,
                    )
                )
                / radius ** (falloff + 1)
            )
            delta_remainder = (
                abs(n_vector[R_INDEX]) * radial_remainder
                + abs(n_vector[THETA_INDEX]) * theta_remainder
            ) / radius**falloff
            angular_vector = m_vector if curvature_index == 0 else mbar_vector
            angular_remainder = (
                abs(angular_vector[R_INDEX]) * radial_remainder
                + abs(angular_vector[THETA_INDEX]) * theta_remainder
            ) / radius ** (falloff + 1)
            maximum_remainder = max(
                maximum_remainder, delta_remainder, angular_remainder
            )
    return tuple(values), tuple(wrong_values), maximum_remainder


def coordinate_scri_estimate(
    spin: float,
    theta: float,
    mode_index: int,
    level: int,
    radial_count: int,
    case_index: int,
    radial_power: int,
) -> tuple[complex, complex]:
    """Extrapolate the already normalized full coordinate-Weyl scalars.

    The six positive-node Lagrange weights extract the constant term of a
    regular scalar with an O(h^6) remainder.  This is intentionally distinct
    from the production q0/q1 numerator moment systems: it neither sees nor
    annihilates the unnormalized peeling residual components.
    """
    spacing = future_horizon_radius(spin) / (radial_count - 1)
    values = []
    for node in range(1, 7):
        radius = node * spacing
        psi0, psi1 = coordinate_modal_curvature(
            spin,
            radius,
            theta,
            mode_index,
            level,
            case_index,
            radial_power,
        )
        values.append((psi0 / radius**5, psi1 / radius**4))
    return tuple(
        sum(
            SCRI_EXTRAPOLATION_WEIGHTS[node] * values[node][field]
            for node in range(6)
        )
        for field in range(2)
    )


def format_complex(value: complex) -> str:
    return (
        "teuk::Complex(" + format(value.real, ".17e") + ", "
        + format(value.imag, ".17e") + ")"
    )


def render() -> str:
    nodes, _weights = np.polynomial.legendre.leggauss(THETA_COUNT)
    theta = [math.acos(float(node)) for node in nodes]
    entries = []
    derivative_entries = []
    scri_entries = []
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
                    if case_index in DERIVATIVE_CASES:
                        for mode_index in range(len(MODES)):
                            derivatives, wrong_derivatives, remainder = (
                                expected_derivatives(
                                    spin,
                                    radial_index,
                                    theta[theta_index],
                                    mode_index,
                                    case_index,
                                )
                            )
                            derivative_entries.append(
                                (
                                    case_index,
                                    spin_index,
                                    radial_index,
                                    theta_index,
                                    mode_index,
                                    derivatives,
                                    wrong_derivatives,
                                    remainder,
                                )
                            )
                            canonical.append(
                                f"d,{case_index},{spin_index},{radial_index},"
                                f"{theta_index},{mode_index}:"
                                + ",".join(
                                    f"{value.real:.17e},{value.imag:.17e}"
                                    for value in derivatives
                                )
                                + ",wrong="
                                + ",".join(
                                    f"{value.real:.17e},{value.imag:.17e}"
                                    for value in wrong_derivatives
                                )
                                + f",remainder={remainder:.17e}"
                            )
                    if (
                        case_index in SCRI_CASES
                        and radial_index == RADIAL_INDICES[0]
                    ):
                        for mode_index in range(len(MODES)):
                            for level in LEVELS:
                                estimates = tuple(
                                    coordinate_scri_estimate(
                                        spin,
                                        theta[theta_index],
                                        mode_index,
                                        level,
                                        radial_count,
                                        case_index,
                                        2,
                                    )
                                    for radial_count in (9, 17, 33)
                                )
                                coarse_medium = tuple(
                                    abs(
                                        estimates[0][field]
                                        - estimates[1][field]
                                    )
                                    for field in range(2)
                                )
                                medium_fine = tuple(
                                    abs(
                                        estimates[1][field]
                                        - estimates[2][field]
                                    )
                                    for field in range(2)
                                )
                                scri_entries.append(
                                    (
                                        case_index,
                                        spin_index,
                                        theta_index,
                                        mode_index,
                                        level,
                                        estimates[2],
                                        coarse_medium,
                                        medium_fine,
                                    )
                                )
                                canonical.append(
                                    f"s,{case_index},{spin_index},"
                                    f"{theta_index},{mode_index},{level}:"
                                    + ",".join(
                                        f"{value.real:.17e},{value.imag:.17e}"
                                        for value in estimates[2]
                                    )
                                    + ",cm="
                                    + ",".join(
                                        f"{value:.17e}" for value in coarse_medium
                                    )
                                    + ",mf="
                                    + ",".join(
                                        f"{value:.17e}" for value in medium_fine
                                    )
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
        f"inline constexpr double derivative_step = {DERIVATIVE_STEP:.17e};",
        "inline constexpr std::array<double, 3> radial_maxes{{"
        + ", ".join(f"{future_horizon_radius(spin):.17e}" for spin in SPINS)
        + "}};",
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
        "struct ScriExpected {",
        "  int case_index;",
        "  int spin_index;",
        "  int theta_index;",
        "  int mode_index;",
        "  int level;",
        "  std::array<teuk::Complex, 2> values;",
        "  std::array<double, 2> coarse_medium;",
        "  std::array<double, 2> medium_fine;",
        "};",
        "",
        f"inline const std::array<ScriExpected, {len(scri_entries)}> "
        "scri_expected{{",
    ])
    for entry in scri_entries:
        ci, si, ti, mi, level, values, coarse_medium, medium_fine = entry
        lines.append(
            f"  {{{ci}, {si}, {ti}, {mi}, {level}, {{{{"
            + ", ".join(format_complex(value) for value in values)
            + "}}, {{"
            + ", ".join(f"{value:.17e}" for value in coarse_medium)
            + "}}, {{"
            + ", ".join(f"{value:.17e}" for value in medium_fine)
            + "}}},"
        )
    lines.extend([
        "}};",
        "",
        "struct DerivativeExpected {",
        "  int case_index;",
        "  int spin_index;",
        "  int radial_index;",
        "  int theta_index;",
        "  int mode_index;",
        "  std::array<teuk::Complex, 8> values;",
        "  std::array<teuk::Complex, 8> wrong_values;",
        "  double finite_difference_remainder;",
        "};",
        "",
        f"inline const std::array<DerivativeExpected, {len(derivative_entries)}> "
        "derivative_expected{{",
    ])
    for entry in derivative_entries:
        ci, si, ri, ti, mi, values, wrong_values, remainder = entry
        lines.append(
            f"  {{{ci}, {si}, {ri}, {ti}, {mi}, {{{{"
            + ", ".join(format_complex(value) for value in values)
            + "}}, {{"
            + ", ".join(format_complex(value) for value in wrong_values)
            + f"}}}}, {remainder:.17e}}},"
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
