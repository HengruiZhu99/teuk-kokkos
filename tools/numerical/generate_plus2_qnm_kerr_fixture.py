#!/usr/bin/env python3
"""Generate a normalized moderate-Kerr spin +2 QNM field fixture.

The complex QNM frequency and spheroidal angular constant are pinned to
``qnm==0.4.4`` for ``s=+2, ell=m=2, n=0, a/M=0.6``.  The package is used only
as an independent spectral reference; this generator does not import it.  It
instead solves the Berens--Gravely--Lupsasca angular Wronskian and horizon-in
radial confluent-Heun problems, then applies the Ripley coordinate/tetrad and
stored-field transformations.

The resulting fixture is an interior homogeneous-mode authority.  It does not
claim a horizon or scri endpoint, an evolved waveform, or a sourced result.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import root

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

import generate_plus2_tsi_kerr_separated_fixture as separated
import generate_plus2_tsi_kerr_t0_fixture as transform

QNM_VERSION = "0.4.4"
QNM_WHEEL_SHA256 = (
    "63711c37ff4847c8cb59e6266da5baf81f88cc6ab728f40fc3fe39a3025f6252"
)
QNM_FREQUENCY = 0.4940447817813845 - 0.0837652021610416j
QNM_ANGULAR_A = -0.8546134005662634 + 0.15669038538845606j
MASS = 1.0
SPIN = 0.6
ELL = 2
MODE = 2
OVERTONE = 0

LEVELS = (
    ("coarse", 1.0e-3, 28, 2.0e-10, 2.0e-12),
    ("medium", 5.0e-4, 36, 2.0e-12, 2.0e-14),
    ("fine", 2.5e-4, 48, 2.0e-13, 2.0e-15),
)
POINTS = ((3.0, 1.1), (4.5, 1.8))


def berens_lambda(omega: complex, m: int, angular_a: complex) -> complex:
    """Convert qnm's angular A to Berens' lambda_plus convention."""
    c = SPIN * omega
    return angular_a - 2.0 * SPIN * m * omega + c * c


def angular_wronskian(
    omega: complex,
    m: int,
    lambda_plus: complex,
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> complex:
    parameters = separated.angular_parameters(
        +2, m, SPIN * omega, lambda_plus
    )
    left = separated.integrate_heun(
        parameters,
        endpoint,
        (0.5,),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )[0]
    right = separated.integrate_heun(
        separated.reflected_heun(parameters),
        endpoint,
        (0.5,),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )[0]
    return left[0] * (-right[1]) - right[0] * left[1]


def solve_angular_lambda(
    omega: complex,
    m: int,
    seed: complex,
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> tuple[complex, float]:
    def residual(parts: np.ndarray) -> np.ndarray:
        value = angular_wronskian(
            omega,
            m,
            complex(parts[0], parts[1]),
            endpoint=endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )
        return np.asarray((value.real, value.imag))

    solution = root(residual, np.asarray((seed.real, seed.imag)), tol=1.0e-12)
    if not solution.success:
        raise AssertionError(f"complex angular root failed: {solution.message}")
    value = complex(solution.x[0], solution.x[1])
    remainder = abs(
        angular_wronskian(
            omega,
            m,
            value,
            endpoint=endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )
    )
    if remainder >= 2.0e-10:
        raise AssertionError(f"complex angular Wronskian is not closed: {remainder:.3e}")
    return value, remainder


def angular_mode(
    fixture: separated.KerrFixture,
    thetas: Sequence[float],
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Return hatted spin+2 S and dS/dtheta with H(0)=1."""
    parameters = separated.angular_parameters(
        +2, fixture.m, fixture.c, fixture.lambda_plus
    )
    z_values = [(1.0 + math.cos(theta)) / 2.0 for theta in thetas]
    ordered = tuple(sorted(z_values))
    values = separated.integrate_heun(
        parameters,
        endpoint,
        ordered,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    by_z = {float(z): value for z, value in zip(ordered, values)}
    result = np.empty((len(thetas), 2), dtype=np.complex128)
    mu1 = abs(2 + fixture.m) / 2.0
    mu2 = abs(2 - fixture.m) / 2.0
    for index, (theta, z) in enumerate(zip(thetas, z_values)):
        h, dh_dz = by_z[float(z)]
        cosine = math.cos(theta)
        prefactor = (
            (1.0 - cosine) ** mu1
            * (1.0 + cosine) ** mu2
            * np.exp(fixture.c * (1.0 + cosine))
        )
        dlog_du = (
            -mu1 / (1.0 - cosine)
            + mu2 / (1.0 + cosine)
            + fixture.c
        )
        result[index, 0] = prefactor * h
        result[index, 1] = -math.sin(theta) * prefactor * (
            dlog_du * h + 0.5 * dh_dz
        )
    return result


def unit_eth(
    value: transform.BiJet, spin: int, m: int, theta: transform.BiJet
) -> transform.BiJet:
    return (
        -value.dtheta()
        + m / theta.sin() * value
        + spin * theta.cos() / theta.sin() * value
    )


def unit_ethprime(
    value: transform.BiJet, spin: int, m: int, theta: transform.BiJet
) -> transform.BiJet:
    return (
        -value.dtheta()
        - m / theta.sin() * value
        - spin * theta.cos() / theta.sin() * value
    )


def field_jets(
    fixture: separated.KerrFixture,
    radius_value: float,
    theta_value: float,
    radial_mode: np.ndarray,
    angular_value: np.ndarray,
) -> tuple[complex, np.ndarray, complex]:
    radial = separated.on_shell_radial_jet(
        fixture,
        +2,
        radius_value,
        radial_mode[0],
        radial_mode[1],
        transform.RADIAL_DEGREE,
    )
    angular = separated.on_shell_angular_jet(
        fixture,
        +2,
        theta_value,
        angular_value[0],
        angular_value[1],
        transform.ANGULAR_DEGREE,
    )
    field = transform.BiJet.outer(radial, angular)
    radius = transform.BiJet.radial(radius_value)
    theta = transform.BiJet.angular(theta_value)
    cosine = theta.cos()
    delta = radius * radius - 2.0 * fixture.mass * radius + fixture.spin**2
    sigma = radius * radius + fixture.spin**2 * cosine * cosine
    boost = delta / (2.0 * sigma)
    q = -(radius + 1j * fixture.spin * cosine) / (
        radius - 1j * fixture.spin * cosine
    )
    height, azimuth = transform.tortoise_and_azimuth(fixture, radius)
    phase = (1j * fixture.omega * height - 1j * fixture.m * azimuth).exp()
    raw = boost * boost * q * q * field * phase

    code_radius = 1.0 / radius
    d_minus = 1.0 - 1j * fixture.spin * code_radius * cosine
    stored = raw * d_minus * d_minus * d_minus * d_minus / (
        code_radius * code_radius * code_radius * code_radius * code_radius
    )
    laplacian = unit_ethprime(
        unit_eth(stored, +2, fixture.m, theta), +3, fixture.m, theta
    )
    stored_code = transform.r_to_code_R(stored, radius_value)
    laplacian_code = transform.r_to_code_R(laplacian, radius_value)
    return raw.c[0, 0], stored_code[:, 0], laplacian_code[0, 0]


def level_data(level: tuple[str, float, int, float, float]) -> list[dict[str, object]]:
    name, endpoint, series_order, rtol, atol = level
    reference_lambda = berens_lambda(QNM_FREQUENCY, MODE, QNM_ANGULAR_A)
    sectors = (
        ("mode", QNM_FREQUENCY, MODE, reference_lambda),
        ("sharp", -np.conj(QNM_FREQUENCY), -MODE, np.conj(reference_lambda)),
    )
    radii = tuple(point[0] for point in POINTS)
    thetas = tuple(point[1] for point in POINTS)
    result: list[dict[str, object]] = []
    for sector, omega, m, seed in sectors:
        lambda_plus, wronskian = solve_angular_lambda(
            complex(omega),
            m,
            complex(seed),
            endpoint=endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )
        fixture = separated.KerrFixture(
            mass=MASS,
            spin=SPIN,
            omega=complex(omega),
            ell=ELL,
            m=m,
            lambda_plus=lambda_plus,
        )
        radial_values = separated.normalized_radial_modes(
            fixture,
            +2,
            radii,
            start=-endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )
        angular_values = angular_mode(
            fixture,
            thetas,
            endpoint=endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )
        for point, ((radius, theta), radial, angular) in enumerate(
            zip(POINTS, radial_values, angular_values)
        ):
            raw, stored, angular_laplacian = field_jets(
                fixture, radius, theta, radial, angular
            )
            result.append(
                {
                    "level": name,
                    "sector": sector,
                    "point": point,
                    "radius": 1.0 / radius,
                    "theta": theta,
                    "omega": complex(omega),
                    "m": m,
                    "lambda_plus": lambda_plus,
                    "wronskian": wronskian,
                    "psi0": raw,
                    "stored": stored,
                    "angular_laplacian": angular_laplacian,
                }
            )
    return result


def maximum_difference(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> float:
    maximum = 0.0
    for lhs, rhs in zip(left, right):
        for key in ("lambda_plus", "psi0", "angular_laplacian"):
            scale = max(abs(rhs[key]), 1.0e-300)  # type: ignore[arg-type]
            maximum = max(
                maximum,
                abs(lhs[key] - rhs[key]) / scale,  # type: ignore[operator]
            )
        lhs_stored = lhs["stored"]  # type: ignore[assignment]
        rhs_stored = rhs["stored"]  # type: ignore[assignment]
        scale = np.maximum(np.abs(rhs_stored), 1.0e-300)
        maximum = max(
            maximum,
            float(np.max(np.abs(lhs_stored - rhs_stored) / scale)),
        )
    return maximum


def c(value: complex) -> str:
    return f"c({value.real:.17e}, {value.imag:.17e})"


def render_header(data: list[dict[str, object]], wrong_wronskian: float) -> str:
    separated_hash = hashlib.sha256(
        (SCRIPT.parent / "generate_plus2_tsi_kerr_separated_fixture.py").read_bytes()
    ).hexdigest()
    transform_hash = hashlib.sha256(
        (SCRIPT.parent / "generate_plus2_tsi_kerr_t0_fixture.py").read_bytes()
    ).hexdigest()
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace teuk::test::plus2_qnm_kerr {",
        "",
        "// Generated file. Do not hand edit.",
        f'inline constexpr const char* generator_sha256 = "{hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* separated_solver_sha256 = "{separated_hash}";',
        f'inline constexpr const char* transform_solver_sha256 = "{transform_hash}";',
        f'inline constexpr const char* qnm_version = "{QNM_VERSION}";',
        f'inline constexpr const char* qnm_wheel_sha256 = "{QNM_WHEEL_SHA256}";',
        f"inline constexpr double wrong_qnm_A_wronskian = {wrong_wronskian:.17e};",
        "struct C { double real; double imag; };",
        "struct Fixture {",
        "  const char* level; const char* sector; int point;",
        "  double radius, theta; C omega; int m; C lambda_plus;",
        "  double wronskian; C psi0_code; std::array<C, 3> stored;",
        "  C angular_laplacian;",
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
                f"      {item['radius']:.17e}, {item['theta']:.17e}, {c(complex(item['omega']))}, {item['m']}, {c(complex(item['lambda_plus']))},",
                f"      {item['wronskian']:.17e}, {c(complex(item['psi0']))}, {{{', '.join(c(complex(value)) for value in stored)}}},",
                f"      {c(complex(item['angular_laplacian']))}",
                "    },",
            ]
        )
    lines.extend(
        ["  };", "}", "", "}  // namespace teuk::test::plus2_qnm_kerr", ""]
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)

    level_values = [level_data(level) for level in LEVELS]
    coarse_medium = maximum_difference(level_values[0], level_values[1])
    medium_fine = maximum_difference(level_values[1], level_values[2])
    reference_lambda = berens_lambda(QNM_FREQUENCY, MODE, QNM_ANGULAR_A)
    fine_lambda = complex(level_values[2][0]["lambda_plus"])
    lambda_error = abs(fine_lambda - reference_lambda)
    wrong_wronskian = abs(
        angular_wronskian(
            QNM_FREQUENCY,
            MODE,
            QNM_ANGULAR_A,
            endpoint=LEVELS[-1][1],
            series_order=LEVELS[-1][2],
            rtol=LEVELS[-1][3],
            atol=LEVELS[-1][4],
        )
    )
    print(
        "spin+2 Kerr QNM fixture convergence "
        f"coarse-medium={coarse_medium:.6e} medium-fine={medium_fine:.6e} "
        f"qnm-lambda-error={lambda_error:.6e} wrong-A-W={wrong_wronskian:.6e}"
    )
    if not medium_fine < coarse_medium or medium_fine >= 3.0e-8:
        raise AssertionError("spin+2 Kerr QNM fixture failed mode convergence")
    if lambda_error >= 2.0e-10:
        raise AssertionError("independent complex angular root misses qnm reference")
    if wrong_wronskian <= 1.0:
        raise AssertionError("wrong qnm angular-A convention was not rejected")
    rendered = render_header(
        [item for level in level_values for item in level], wrong_wronskian
    )
    if args.output:
        args.output.write_text(rendered)
    elif args.check:
        if args.check.read_text() != rendered:
            raise AssertionError(f"generated fixture differs from {args.check}")
        print(f"PASS generated spin+2 Kerr QNM fixture matches {args.check}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
