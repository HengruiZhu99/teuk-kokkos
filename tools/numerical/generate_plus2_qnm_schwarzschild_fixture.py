#!/usr/bin/env python3
"""Generate a normalized Schwarzschild spin +2 QNM field fixture.

The radial function is the horizon-in solution of Berens--Gravely--Lupsasca
arXiv:2403.20311 Eqs. (2.17)--(2.18), with the local confluent-Heun series
normalized to unit leading coefficient.  The QNM frequency is independently
pinned by qnm 0.4.4 (Leaver/Cook--Zalutskiy), evaluated at s=+2, l=m=2,
n=0, a=0.  The transform to the repository's horizon-regular tetrad,
hyperboloidal time, compact radius R=1/r, and stored Z_plus follows Ripley et
al. arXiv:2010.00162 Eqs. (5)--(6), (21b), and Appendix C.

This generator imports no production C++ algebra.  It validates an interior,
homogeneous, separated QNM field; it does not claim a horizon/scri endpoint,
an evolved waveform, Kerr spin, or a quadratic-source result.
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

import generate_plus2_tsi_schwarzschild_t0_fixture as transform
import verify_plus2_tsi_schwarzschild_radial_fixture as radial

BERENS_ARXIV_TEX_SHA256 = transform.BERENS_ARXIV_TEX_SHA256
BERENS_SUPPLEMENT_COMMIT = transform.BERENS_SUPPLEMENT_COMMIT
BERENS_EXAMPLE_NB_SHA256 = transform.BERENS_EXAMPLE_NB_SHA256
RIPLEY_ARXIV_TEX_SHA256 = transform.RIPLEY_ARXIV_TEX_SHA256
QNM_VERSION = "0.4.4"
QNM_WHEEL_SHA256 = (
    "63711c37ff4847c8cb59e6266da5baf81f88cc6ab728f40fc3fe39a3025f6252"
)
QNM_FREQUENCY = 0.373671684418042 - 0.08896231568893723j

LEVELS = (
    (
        "coarse",
        {"start": -4.0e-3, "series_order": 20, "rtol": 2.0e-9, "atol": 2.0e-11},
    ),
    (
        "medium",
        {"start": -4.0e-3, "series_order": 32, "rtol": 2.0e-11, "atol": 2.0e-13},
    ),
    (
        "fine",
        {"start": -4.0e-3, "series_order": 48, "rtol": 2.3e-14, "atol": 2.0e-16},
    ),
)
POINTS = ((4.0, 0.9), (6.0, 1.7))


def qnm_fixture(omega: complex, m: int) -> radial.Fixture:
    return radial.Fixture(omega=omega, m=m)


def spin_plus_angular(theta: float, m: int) -> complex:
    """Unnormalized Schwarzschild {}_2Y_{2m} polynomial with unit coefficient."""
    if m == 2:
        return complex((1.0 - np.cos(theta)) ** 2)
    if m == -2:
        return complex((1.0 + np.cos(theta)) ** 2)
    raise ValueError("fixture supports only m=+/-2")


def radial_code_jets(
    fixture: radial.Fixture, radius: float, mode: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Taylor jets in code R for Psi0_code and stored Z_plus."""
    degree = 2
    code_radius = radial.jet_variable(1.0 / radius, degree)
    bl_radius = radial.jet_inverse(code_radius)
    radial_increment = bl_radius - radius * radial.jet_constant(1.0, degree)
    mode_r = radial.radial_solution_jet(
        fixture, +2, radius, mode[0], mode[1], degree
    )
    mode_R = transform.compose_taylor(mode_r, radial_increment)
    phase = transform.exp_jet(
        1j * fixture.omega * transform.schwarzschild_height(bl_radius, fixture.mass)
    )
    delta = transform.delta_jet(bl_radius, fixture.mass)
    boost = radial.jet_divide(
        delta, 2.0 * radial.jet_multiply(bl_radius, bl_radius)
    )
    psi0 = radial.jet_multiply(
        radial.jet_multiply(radial.jet_multiply(boost, boost), mode_R), phase
    )
    radius2 = radial.jet_multiply(code_radius, code_radius)
    radius4 = radial.jet_multiply(radius2, radius2)
    stored = radial.jet_divide(psi0, radial.jet_multiply(radius4, code_radius))
    return psi0, stored


def normalized_data(
    level: str, options: dict[str, float | int]
) -> list[dict[str, object]]:
    sectors = (
        ("mode", QNM_FREQUENCY, 2),
        ("sharp", -np.conj(QNM_FREQUENCY), -2),
    )
    radii = tuple(point[0] for point in POINTS)
    result: list[dict[str, object]] = []
    mode_values: dict[str, np.ndarray] = {}
    for name, omega, m in sectors:
        fixture = qnm_fixture(complex(omega), m)
        mode_values[name] = radial.normalized_radial_modes(
            fixture, +2, radii, **options
        )
        for point, ((radius_value, theta), mode) in enumerate(
            zip(POINTS, mode_values[name])
        ):
            psi0_radial, stored_radial = radial_code_jets(
                fixture, radius_value, mode
            )
            angular = spin_plus_angular(theta, m)
            result.append(
                {
                    "level": level,
                    "sector": name,
                    "point": point,
                    "radius": 1.0 / radius_value,
                    "theta": theta,
                    "omega": complex(omega),
                    "m": m,
                    "psi0": psi0_radial[0] * angular,
                    "stored": stored_radial * angular,
                }
            )

    symmetry = np.max(
        np.abs(mode_values["sharp"] - np.conj(mode_values["mode"]))
        / np.maximum(np.abs(mode_values["mode"]), 1.0e-300)
    )
    if symmetry >= 3.0e-10:
        raise AssertionError(f"QNM sharp radial symmetry failed: {symmetry:.3e}")
    return result


def maximum_difference(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> float:
    maximum = 0.0
    for lhs, rhs in zip(left, right):
        for key in ("psi0",):
            scale = max(abs(rhs[key]), 1.0e-300)  # type: ignore[arg-type]
            maximum = max(maximum, abs(lhs[key] - rhs[key]) / scale)  # type: ignore[operator]
        lhs_stored = lhs["stored"]  # type: ignore[assignment]
        rhs_stored = rhs["stored"]  # type: ignore[assignment]
        scale = np.maximum(np.abs(rhs_stored), 1.0e-300)
        maximum = max(maximum, float(np.max(np.abs(lhs_stored - rhs_stored) / scale)))
    return maximum


def c(value: complex) -> str:
    return f"c({value.real:.17e}, {value.imag:.17e})"


def render_header(data: list[dict[str, object]]) -> str:
    radial_script = SCRIPT.parent / "verify_plus2_tsi_schwarzschild_radial_fixture.py"
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace teuk::test::plus2_qnm_schwarzschild {",
        "",
        "// Generated file. Do not hand edit.",
        f'inline constexpr const char* generator_sha256 = "{hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* radial_solver_sha256 = "{hashlib.sha256(radial_script.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* qnm_version = "{QNM_VERSION}";',
        f'inline constexpr const char* qnm_wheel_sha256 = "{QNM_WHEEL_SHA256}";',
        f'inline constexpr const char* berens_arxiv_tex_sha256 = "{BERENS_ARXIV_TEX_SHA256}";',
        f'inline constexpr const char* berens_supplement_commit = "{BERENS_SUPPLEMENT_COMMIT}";',
        f'inline constexpr const char* berens_example_nb_sha256 = "{BERENS_EXAMPLE_NB_SHA256}";',
        f'inline constexpr const char* ripley_arxiv_tex_sha256 = "{RIPLEY_ARXIV_TEX_SHA256}";',
        "struct C { double real; double imag; };",
        "struct Fixture {",
        "  const char* level; const char* sector; int point;",
        "  double radius, theta; C omega; int m;",
        "  C psi0_code; std::array<C, 3> stored; // Z, d_R Z, d_RR Z/2",
        "};",
        "constexpr C c(double real, double imag) { return {real, imag}; }",
        f"inline std::array<Fixture, {len(data)}> fixtures() {{",
        "  return {",
    ]
    for item in data:
        stored = item["stored"]
        lines.extend(
            [
                "    Fixture{",
                f'      "{item["level"]}", "{item["sector"]}", {item["point"]},',
                f"      {item['radius']:.17e}, {item['theta']:.17e}, {c(complex(item['omega']))}, {item['m']},",
                f"      {c(complex(item['psi0']))}, {{{', '.join(c(complex(value)) for value in stored)}}}",
                "    },",
            ]
        )
    lines.extend(["  };", "}", "", "}  // namespace teuk::test::plus2_qnm_schwarzschild", ""])
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    level_data = [normalized_data(name, options) for name, options in LEVELS]
    coarse_medium = maximum_difference(level_data[0], level_data[1])
    medium_fine = maximum_difference(level_data[1], level_data[2])
    print(
        f"spin+2 QNM fixture convergence coarse-medium={coarse_medium:.6e} "
        f"medium-fine={medium_fine:.6e}"
    )
    if not medium_fine < coarse_medium or medium_fine >= 2.0e-8:
        raise AssertionError("spin+2 QNM fixture failed normalized-mode convergence")
    rendered = render_header([item for level in level_data for item in level])
    if args.output:
        args.output.write_text(rendered)
    elif args.check:
        if args.check.read_text() != rendered:
            raise AssertionError(f"generated fixture differs from {args.check}")
        print(f"PASS generated spin+2 QNM fixture matches {args.check}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
