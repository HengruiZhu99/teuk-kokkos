#!/usr/bin/env python3
"""Generate an independently normalized Schwarzschild T0/TSI fixture.

The metric comes from Berens--Gravely--Lupsasca arXiv:2403.20311
Eqs. (2.44), (2.50), and (2.52).  The hatted radial modes come from the
neighboring numerical Heun/RTSI verifier.  The conversion to the repository
tetrad and coordinates follows Ripley et al. arXiv:2010.00162 Eqs. (5)--(6)
and Appendix C.  No production T0 formula is imported or transcribed here.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from verify_plus2_tsi_schwarzschild_radial_fixture import (
    Fixture,
    jet_constant,
    jet_derivative,
    jet_divide,
    jet_inverse,
    jet_multiply,
    jet_variable,
    normalized_radial_modes,
    radial_operator,
    radial_solution_jet,
)

# Hashes of the exact primary sources used to derive this generator.
BERENS_ARXIV_TEX_SHA256 = (
    "f5af07f8692e4f8d49f271e41056eadb0bf180e78a6c8574f238930f2298b3cf"
)
BERENS_SUPPLEMENT_COMMIT = "cf924707593a58ec889c70ea501d764e99d1d4aa"
BERENS_EXAMPLE_NB_SHA256 = (
    "d2c69c27f34c4da852c355a3240a1f351c05340182ee740bdf8fa56e5c05c974"
)
RIPLEY_ARXIV_TEX_SHA256 = (
    "2518ef1168e552db4ca4fd07ee421fca33f0939ecfc2031d272842b61cbf955e"
)

LEVELS = (
    (
        "coarse",
        {"start": -2.0e-4, "series_order": 24, "rtol": 2.0e-10, "atol": 2.0e-12},
    ),
    (
        "medium",
        {"start": -1.0e-4, "series_order": 32, "rtol": 2.0e-12, "atol": 2.0e-14},
    ),
    (
        "fine",
        {"start": -5.0e-5, "series_order": 40, "rtol": 2.0e-13, "atol": 2.0e-15},
    ),
)
POINTS = ((4.0, 1.1), (6.0, 0.8))


def exp_jet(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    result[0] = np.exp(value[0])
    for n in range(1, len(value)):
        result[n] = sum(k * value[k] * result[n - k] for k in range(1, n + 1)) / n
    return result


def log_jet(value: np.ndarray) -> np.ndarray:
    quotient = jet_divide(jet_derivative(value), value[:-1])
    result = np.zeros_like(value)
    result[0] = np.log(value[0])
    for n, coefficient in enumerate(quotient):
        result[n + 1] = coefficient / (n + 1)
    return result


def cos_jet(value: np.ndarray) -> np.ndarray:
    return 0.5 * (exp_jet(1j * value) + exp_jet(-1j * value))


def sin_jet(value: np.ndarray) -> np.ndarray:
    return (exp_jet(1j * value) - exp_jet(-1j * value)) / (2j)


def compose_taylor(outer: np.ndarray, inner_delta: np.ndarray) -> np.ndarray:
    result = np.zeros_like(inner_delta)
    power = jet_constant(1.0, len(inner_delta) - 1)
    for coefficient in outer:
        result += coefficient * power
        power = jet_multiply(power, inner_delta)
    return result


def angular_operator(
    value: np.ndarray, theta: np.ndarray, m: int, n: int
) -> np.ndarray:
    algebraic = jet_divide(
        m * jet_constant(1.0, len(theta) - 1) + n * cos_jet(theta),
        sin_jet(theta),
    )
    return jet_derivative(value) + jet_multiply(algebraic, value)[:-1]


def delta_jet(radius: np.ndarray, mass: float) -> np.ndarray:
    return jet_multiply(
        radius, radius - 2.0 * mass * jet_constant(1.0, len(radius) - 1)
    )


def schwarzschild_height(radius: np.ndarray, mass: float) -> np.ndarray:
    """T-t from Ripley Eqs. (C4), (C15), and r*=r+2M log(...)."""
    degree = len(radius) - 1
    one = jet_constant(1.0, degree)
    return (
        -radius
        + 2.0 * mass * log_jet((radius - 2.0 * mass * one) / (2.0 * mass))
        - 4.0 * mass * log_jet(radius)
    )


def check_coordinate_and_tetrad_chain() -> None:
    """Check Jacobian signs against both paper tetrads, independently of T0."""
    mass = 1.0
    for radius in (2.5, 4.0, 9.0):
        delta = radius * (radius - 2.0 * mass)
        boost = delta / (2.0 * radius * radius)
        h_prime = -1.0 + 2.0 * mass / (radius - 2.0 * mass) - 4.0 * mass / radius
        rstar_prime = radius / (radius - 2.0 * mass)
        assert abs(h_prime - (rstar_prime - 2.0 - 4.0 * mass / radius)) < 2.0e-15

        # Transform the BL Kinnersley l,n through T=t+H(r), R=1/r and
        # then apply l_code=A_b l_Kin, n_code=A_b^-1 n_Kin.
        l_t = radius * radius / delta
        l_r = 1.0
        transformed_l_t = boost * (l_t + h_prime * l_r)
        transformed_l_r = boost * (-l_r / radius**2)
        assert abs(transformed_l_t - 4.0 * mass * mass / radius**2) < 2.0e-15
        assert (
            abs(transformed_l_r + (1.0 - 2.0 * mass / radius) / (2.0 * radius**2))
            < 2.0e-15
        )

        n_t = 0.5
        n_r = -delta / (2.0 * radius * radius)
        transformed_n_t = (n_t + h_prime * n_r) / boost
        transformed_n_r = (-n_r / radius**2) / boost
        assert abs(transformed_n_t - (2.0 + 4.0 * mass / radius)) < 2.0e-15
        assert abs(transformed_n_r - 1.0 / radius**2) < 2.0e-15

    # q=-(r+i a cos theta)/(r-i a cos theta) is exactly -1 at a=0.
    for theta in (0.4, 1.1, 2.2):
        q = -(4.0 + 0j * np.cos(theta)) / (4.0 - 0j * np.cos(theta))
        assert q == -1.0 + 0j
    print("PASS Ripley height-function signs and Schwarzschild tetrad Jacobian")


def partner_h_radial(
    fixture: Fixture, radius: float, radial_mode: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Berens Eq. (2.52) at a=0 and epsilon_g=-1."""
    radial = radial_solution_jet(fixture, +2, radius, radial_mode[0], radial_mode[1], 4)
    r = jet_variable(radius, 4)
    delta = delta_jet(r, fixture.mass)
    inner = jet_multiply(jet_multiply(delta, delta), radial)
    first = radial_operator(fixture, radius, inner, dagger=True)
    shifted_first = first - 2.0 * jet_divide(inner[: len(first)], r[: len(first)])
    second = radial_operator(fixture, radius, first, dagger=True)
    second -= 2.0 * jet_divide(first[: len(second)], r[: len(second)])
    h_ll = 0.25 * jet_multiply(jet_multiply(r, r), radial)
    h_lm = -(1.0 / (4.0 * np.sqrt(2.0))) * jet_multiply(
        jet_divide(r[: len(shifted_first)], delta[: len(shifted_first)]),
        shifted_first,
    )
    h_mm = 0.125 * second
    return h_ll[:3], h_lm[:3], h_mm[:3]


def convert_radial_to_code(
    fixture: Fixture,
    radius: float,
    omega: float,
    radial_factors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = 2
    code_radius = jet_variable(1.0 / radius, degree)
    bl_radius = jet_inverse(code_radius)
    radial_increment = bl_radius - radius * jet_constant(1.0, degree)
    phase = exp_jet(1j * omega * schwarzschild_height(bl_radius, fixture.mass))
    delta = delta_jet(bl_radius, fixture.mass)
    boost = jet_divide(delta, 2.0 * jet_multiply(bl_radius, bl_radius))
    # h_ll, h_lm, h_mm transform with A_b^2, A_b q, q^2; q=-1.
    tetrad_factors = (jet_multiply(boost, boost), -boost, jet_constant(1.0, degree))
    return tuple(
        jet_multiply(
            jet_multiply(compose_taylor(factor, radial_increment), tetrad),
            phase,
        )
        for factor, tetrad in zip(radial_factors, tetrad_factors)
    )  # type: ignore[return-value]


def angular_factors(theta_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = jet_variable(theta_value, 4)
    s_partner = jet_multiply(
        jet_constant(1.0, 4) + cos_jet(theta),
        jet_constant(1.0, 4) + cos_jet(theta),
    )
    l2 = angular_operator(s_partner, theta, -2, 2)
    l1_l2 = angular_operator(l2, theta[: len(l2)], -2, 1)
    return l1_l2[:3], l2[:3], s_partner[:3]


def radial_constants(fixture: Fixture) -> tuple[complex, complex, complex]:
    sigma = fixture.sigma
    w = 4.0 * fixture.mass * fixture.omega * fixture.r_plus
    gamma = (w + 2j * sigma) * (w + 1j * sigma) * w * (w - 1j * sigma)
    d_product = float(
        ((fixture.ell - 1) * fixture.ell * (fixture.ell + 1) * (fixture.ell + 2)) ** 2
    )
    c_product = complex(d_product + (12.0 * fixture.mass * fixture.omega) ** 2)
    return gamma, c_product / gamma, c_product


def check_normalization_and_phase(
    original: Fixture, original_modes: np.ndarray, partner_modes: np.ndarray
) -> None:
    gamma, c_prime, c_product = radial_constants(original)
    assert abs(gamma * c_prime - c_product) < 2.0e-13
    symmetry = np.max(
        np.abs(partner_modes - np.conj(original_modes))
        / np.maximum(np.abs(original_modes), 1.0e-300)
    )
    if symmetry >= 2.0e-10:
        raise AssertionError(f"signed hatted radial symmetry failed: {symmetry:.3e}")
    same = 96.0 / c_prime
    sharp = 48j * original.omega * original.mass / np.conj(c_prime)
    expected_ratio = 0.5j * original.omega * c_prime / np.conj(c_prime)
    if abs(sharp / same - expected_ratio) >= 2.0e-15:
        raise AssertionError("Eq. (2.44) signed phase ratio lost a conjugation")
    print(
        "PASS hatted radial symmetry and Eq.2.44 signed phase "
        f"ratio={sharp / same:.17g}"
    )


def normalized_data(
    level: str, options: dict[str, float | int]
) -> list[dict[str, object]]:
    original = Fixture()
    partner = Fixture(omega=-original.omega, m=-original.m)
    radii = tuple(point[0] for point in POINTS)
    original_plus = normalized_radial_modes(original, +2, radii, **options)
    partner_plus = normalized_radial_modes(partner, +2, radii, **options)
    check_normalization_and_phase(original, original_plus, partner_plus)
    _, c_prime, _ = radial_constants(original)
    b_org = 64.0 / np.conj(c_prime)
    same_amplitude = 96.0 / c_prime
    sharp_amplitude = 48j * original.omega * original.mass / np.conj(c_prime)
    result: list[dict[str, object]] = []
    for point_index, ((radius, theta), original_mode, partner_mode) in enumerate(
        zip(POINTS, original_plus, partner_plus)
    ):
        h_partner = partner_h_radial(partner, radius, partner_mode)
        angular_partner = angular_factors(theta)
        minus_code = convert_radial_to_code(
            original,
            radius,
            -original.omega,
            tuple(b_org * value for value in h_partner),
        )
        # Berens Eq. (2.50): A_ORG=0, so only h_ll survives in +m.
        plus_code = convert_radial_to_code(
            original,
            radius,
            original.omega,
            (
                np.conj(b_org) * np.conj(h_partner[0]),
                np.zeros(3, dtype=np.complex128),
                np.zeros(3, dtype=np.complex128),
            ),
        )
        boost = (radius - 2.0 * original.mass) / (2.0 * radius)
        height = (
            -radius
            + 2.0
            * original.mass
            * np.log((radius - 2.0 * original.mass) / (2.0 * original.mass))
            - 4.0 * original.mass * np.log(radius)
        )
        expected_plus = (
            boost**2
            * same_amplitude
            * original_mode[0]
            * (1.0 - np.cos(theta)) ** 2
            * np.exp(1j * original.omega * height)
        )
        expected_minus = (
            boost**2
            * sharp_amplitude
            * partner_mode[0]
            * (1.0 + np.cos(theta)) ** 2
            * np.exp(-1j * original.omega * height)
        )
        common = {
            "level": level,
            "point": point_index,
            "radius": 1.0 / radius,
            "theta": theta,
        }
        result.extend(
            (
                common
                | {
                    "sector": "same_mode",
                    "omega": original.omega,
                    "m": original.m,
                    "radial": plus_code,
                    "angular": (angular_partner[0], np.zeros(3), np.zeros(3)),
                    "expected": expected_plus,
                },
                common
                | {
                    "sector": "sharp_partner",
                    "omega": -original.omega,
                    "m": -original.m,
                    "radial": minus_code,
                    "angular": angular_partner,
                    "expected": expected_minus,
                },
            )
        )
    return result


def maximum_difference(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> float:
    maximum = 0.0
    for lhs, rhs in zip(left, right):
        for key in ("radial", "angular"):
            for lhs_component, rhs_component in zip(lhs[key], rhs[key]):  # type: ignore[index]
                scale = np.maximum(np.abs(rhs_component), 1.0e-300)
                maximum = max(
                    maximum,
                    float(np.max(np.abs(lhs_component - rhs_component) / scale)),
                )
        scale = max(abs(rhs["expected"]), 1.0e-300)  # type: ignore[arg-type]
        maximum = max(
            maximum,
            abs(lhs["expected"] - rhs["expected"]) / scale,  # type: ignore[operator]
        )
    return maximum


def c(value: complex) -> str:
    return f"c({value.real:.17e}, {value.imag:.17e})"


def separated(radial: np.ndarray, angular: np.ndarray) -> str:
    return (
        "component({"
        + ", ".join(c(complex(value)) for value in radial)
        + "}, {"
        + ", ".join(c(complex(value)) for value in angular)
        + "})"
    )


def render_header(data: list[dict[str, object]]) -> str:
    radial_script = SCRIPT.parent / "verify_plus2_tsi_schwarzschild_radial_fixture.py"
    generator_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    radial_hash = hashlib.sha256(radial_script.read_bytes()).hexdigest()
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace teuk::test::plus2_tsi {",
        "",
        "// Generated file. Do not hand edit.",
        f'inline constexpr const char* generator_sha256 = "{generator_hash}";',
        f'inline constexpr const char* radial_solver_sha256 = "{radial_hash}";',
        f'inline constexpr const char* berens_arxiv_tex_sha256 = "{BERENS_ARXIV_TEX_SHA256}";',
        f'inline constexpr const char* berens_supplement_commit = "{BERENS_SUPPLEMENT_COMMIT}";',
        f'inline constexpr const char* berens_example_nb_sha256 = "{BERENS_EXAMPLE_NB_SHA256}";',
        f'inline constexpr const char* ripley_arxiv_tex_sha256 = "{RIPLEY_ARXIV_TEX_SHA256}";',
        "",
        "struct C { double real; double imag; };",
        "struct SeparatedComponent {",
        "  std::array<C, 3> radial;  // value, d_R, d_RR/2",
        "  std::array<C, 3> angular; // value, d_theta, d_thetatheta/2",
        "};",
        "struct T0Fixture {",
        "  const char* level;",
        "  const char* sector;",
        "  int point;",
        "  double radius;",
        "  double theta;",
        "  double omega;",
        "  int m;",
        "  std::array<SeparatedComponent, 3> metric;",
        "  C expected_psi0_code;",
        "};",
        "constexpr C c(double real, double imag) { return {real, imag}; }",
        "constexpr SeparatedComponent component(const std::array<C, 3>& radial,",
        "                                       const std::array<C, 3>& angular) {",
        "  return {radial, angular};",
        "}",
        f"inline std::array<T0Fixture, {len(data)}> schwarzschild_t0_fixtures() {{",
        "  return {",
    ]
    for item in data:
        components = [
            separated(radial, angular)
            for radial, angular in zip(item["radial"], item["angular"])  # type: ignore[arg-type]
        ]
        lines.extend(
            [
                "    T0Fixture{",
                f'      "{item["level"]}", "{item["sector"]}", {item["point"]},',
                f"      {item['radius']:.17e}, {item['theta']:.17e},",
                f"      {item['omega']:.17e}, {item['m']},",
                "      std::array<SeparatedComponent, 3>{",
                "        " + ",\n        ".join(components),
                "      },",
                f"      {c(complex(item['expected']))}",
                "    },",
            ]
        )
    lines.extend(["  };", "}", "", "}  // namespace teuk::test::plus2_tsi", ""])
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    check_coordinate_and_tetrad_chain()
    level_data = [normalized_data(name, options) for name, options in LEVELS]
    coarse_medium = maximum_difference(level_data[0], level_data[1])
    medium_fine = maximum_difference(level_data[1], level_data[2])
    print(
        f"T0 fixture convergence coarse-medium={coarse_medium:.6e} "
        f"medium-fine={medium_fine:.6e}"
    )
    if not medium_fine < coarse_medium or medium_fine >= 5.0e-10:
        raise AssertionError("T0 fixture failed normalized-mode convergence gate")
    rendered = render_header([item for level in level_data for item in level])
    if args.output:
        args.output.write_text(rendered)
    elif args.check:
        if args.check.read_text() != rendered:
            raise AssertionError(f"generated fixture differs from {args.check}")
        print(f"PASS generated T0 fixture matches {args.check}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
