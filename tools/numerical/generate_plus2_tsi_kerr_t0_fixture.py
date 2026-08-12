#!/usr/bin/env python3
"""Generate normalized moderate-Kerr ORG metric/T0 fixtures.

The hatted modes and Starobinsky factors come from the independent separated
solver in ``generate_plus2_tsi_kerr_separated_fixture.py``.  This script then
applies Berens--Gravely--Lupsasca arXiv:2403.20311 Eqs. (2.44), (2.50), and
(2.52), followed by the coordinate/tetrad transformation of Ripley et al.
arXiv:2010.00162 Eqs. (5)--(6) and Appendix C.  It never imports or transcribes
the production ``evaluate_plus2_linear_psi0`` algebra.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import generate_plus2_tsi_kerr_separated_fixture as separated
import generate_plus2_tsi_schwarzschild_t0_fixture as schwarzschild

BERENS_ARXIV_TEX_SHA256 = separated.BERENS_ARXIV_TEX_SHA256
BERENS_SUPPLEMENT_COMMIT = separated.BERENS_SUPPLEMENT_COMMIT
BERENS_EXAMPLE_NB_SHA256 = separated.BERENS_EXAMPLE_NB_SHA256
RIPLEY_ARXIV_TEX_SHA256 = (
    "2518ef1168e552db4ca4fd07ee421fca33f0939ecfc2031d272842b61cbf955e"
)

POINTS = ((3.0, 1.1), (4.5, 1.8))
RADIAL_DEGREE = 4
ANGULAR_DEGREE = 4


class BiJet:
    """Rectangular bivariate Taylor coefficients in (r-r0, theta-theta0)."""

    __array_priority__ = 1000

    def __init__(self, coefficients: np.ndarray):
        self.c = np.asarray(coefficients, dtype=np.complex128)

    @staticmethod
    def constant(value: complex) -> "BiJet":
        coefficients = np.zeros(
            (RADIAL_DEGREE + 1, ANGULAR_DEGREE + 1), dtype=np.complex128
        )
        coefficients[0, 0] = value
        return BiJet(coefficients)

    @staticmethod
    def radial(value: float) -> "BiJet":
        result = BiJet.constant(value)
        result.c[1, 0] = 1.0
        return result

    @staticmethod
    def angular(value: float) -> "BiJet":
        result = BiJet.constant(value)
        result.c[0, 1] = 1.0
        return result

    @staticmethod
    def outer(radial: np.ndarray, angular: np.ndarray) -> "BiJet":
        coefficients = np.zeros(
            (RADIAL_DEGREE + 1, ANGULAR_DEGREE + 1), dtype=np.complex128
        )
        coefficients[: len(radial), : len(angular)] = np.outer(radial, angular)
        return BiJet(coefficients)

    def __add__(self, other: object) -> "BiJet":
        rhs = other if isinstance(other, BiJet) else BiJet.constant(complex(other))
        return BiJet(self.c + rhs.c)

    __radd__ = __add__

    def __neg__(self) -> "BiJet":
        return BiJet(-self.c)

    def __sub__(self, other: object) -> "BiJet":
        return self + (-other if isinstance(other, BiJet) else -complex(other))

    def __rsub__(self, other: object) -> "BiJet":
        return BiJet.constant(complex(other)) - self

    def __mul__(self, other: object) -> "BiJet":
        rhs = other if isinstance(other, BiJet) else BiJet.constant(complex(other))
        result = np.zeros_like(self.c)
        for i in range(RADIAL_DEGREE + 1):
            for j in range(ANGULAR_DEGREE + 1):
                result[i, j] = sum(
                    self.c[k, ell] * rhs.c[i - k, j - ell]
                    for k in range(i + 1)
                    for ell in range(j + 1)
                )
        return BiJet(result)

    __rmul__ = __mul__

    def reciprocal(self) -> "BiJet":
        if self.c[0, 0] == 0.0:
            raise ZeroDivisionError("cannot invert a jet with zero constant")
        constant = self.c[0, 0]
        remainder = self / constant - 1.0
        result = BiJet.constant(1.0)
        power = BiJet.constant(1.0)
        for order in range(1, RADIAL_DEGREE + ANGULAR_DEGREE + 1):
            power = power * remainder
            result = result + ((-1) ** order) * power
        return result / constant

    def __truediv__(self, other: object) -> "BiJet":
        if isinstance(other, BiJet):
            return self * other.reciprocal()
        return BiJet(self.c / complex(other))

    def __rtruediv__(self, other: object) -> "BiJet":
        return BiJet.constant(complex(other)) * self.reciprocal()

    def dr(self) -> "BiJet":
        result = np.zeros_like(self.c)
        for i in range(RADIAL_DEGREE):
            result[i, :] = (i + 1) * self.c[i + 1, :]
        return BiJet(result)

    def dtheta(self) -> "BiJet":
        result = np.zeros_like(self.c)
        for j in range(ANGULAR_DEGREE):
            result[:, j] = (j + 1) * self.c[:, j + 1]
        return BiJet(result)

    def exp(self) -> "BiJet":
        constant = self.c[0, 0]
        remainder = self - constant
        result = BiJet.constant(1.0)
        power = BiJet.constant(1.0)
        factorial = 1.0
        for order in range(1, RADIAL_DEGREE + ANGULAR_DEGREE + 1):
            power = power * remainder
            factorial *= order
            result = result + power / factorial
        return np.exp(constant) * result

    def log(self) -> "BiJet":
        constant = self.c[0, 0]
        remainder = self / constant - 1.0
        result = BiJet.constant(np.log(constant))
        power = BiJet.constant(1.0)
        for order in range(1, RADIAL_DEGREE + ANGULAR_DEGREE + 1):
            power = power * remainder
            result = result + ((-1) ** (order + 1)) * power / order
        return result

    def sin(self) -> "BiJet":
        return ((1j * self).exp() - (-1j * self).exp()) / (2j)

    def cos(self) -> "BiJet":
        return ((1j * self).exp() + (-1j * self).exp()) / 2.0

    def conj(self) -> "BiJet":
        return BiJet(np.conj(self.c))


def radial_operator(
    fixture: separated.KerrFixture, radius: BiJet, value: BiJet, *, dagger: bool
) -> BiJet:
    delta = radius * radius - 2.0 * fixture.mass * radius + fixture.spin**2
    k = fixture.omega * (radius * radius + fixture.spin**2) - fixture.spin * fixture.m
    return value.dr() + (1j if dagger else -1j) * k / delta * value


def angular_operator(
    fixture: separated.KerrFixture,
    theta: BiJet,
    value: BiJet,
    n: int,
    *,
    dagger: bool = False,
) -> BiJet:
    sine = theta.sin()
    q = -fixture.c * sine + fixture.m / sine
    algebraic = (-q if dagger else q) + n * theta.cos() / sine
    return value.dtheta() + algebraic * value


def partner_org_h(
    fixture: separated.KerrFixture,
    radius_value: float,
    theta_value: float,
    radial_mode: np.ndarray,
    angular_mode: np.ndarray,
) -> tuple[BiJet, BiJet, BiJet]:
    """Berens Eq. (2.52), epsilon_g=-1, for one hatted +2 mode."""
    radial = separated.on_shell_radial_jet(
        fixture,
        2,
        radius_value,
        radial_mode[0],
        radial_mode[1],
        RADIAL_DEGREE,
    )
    angular = separated.on_shell_angular_jet(
        fixture,
        2,
        theta_value,
        angular_mode[0],
        angular_mode[1],
        ANGULAR_DEGREE,
    )
    field = BiJet.outer(radial, angular)
    radius = BiJet.radial(radius_value)
    theta = BiJet.angular(theta_value)
    sine = theta.sin()
    cosine = theta.cos()
    zeta = radius - 1j * fixture.spin * cosine
    zeta_bar = radius + 1j * fixture.spin * cosine
    delta = radius * radius - 2.0 * fixture.mass * radius + fixture.spin**2
    sigma = radius * radius + fixture.spin**2 * cosine * cosine

    l2 = angular_operator(fixture, theta, field, 2)
    h_ll = 0.25 * zeta * zeta * (
        angular_operator(fixture, theta, l2, 1)
        - 2j * fixture.spin * sine / zeta * l2
    )

    delta2_field = delta * delta * field
    l2_delta2 = angular_operator(fixture, theta, delta2_field, 2)
    d_delta2 = radial_operator(fixture, radius, delta2_field, dagger=True)
    h_lm = -(1.0 / (4.0 * math.sqrt(2.0))) * zeta * zeta / (zeta_bar * delta) * (
        radial_operator(fixture, radius, l2_delta2, dagger=True)
        + fixture.spin**2 * (2.0 * theta).sin() / sigma * d_delta2
        - 2.0 * radius / sigma * l2_delta2
    )

    first_d = d_delta2
    h_mm = 0.125 * zeta * zeta / (zeta_bar * zeta_bar) * (
        radial_operator(fixture, radius, first_d, dagger=True)
        - 2.0 / zeta * first_d
    )
    return h_ll, h_lm, h_mm


def tortoise_and_azimuth(
    fixture: separated.KerrFixture, radius: BiJet
) -> tuple[BiJet, BiJet]:
    """Fix integration constants and return H=T-t and J=phi_code-phi_BL."""
    rp = fixture.r_plus
    rm = fixture.r_minus
    sigma = fixture.sigma
    scale = 2.0 * fixture.mass
    tortoise = (
        radius
        + 2.0 * fixture.mass * rp / sigma * ((radius - rp) / scale).log()
        - 2.0 * fixture.mass * rm / sigma * ((radius - rm) / scale).log()
    )
    height = tortoise - 2.0 * radius - 4.0 * fixture.mass * radius.log()
    azimuth = fixture.spin / sigma * ((radius - rp) / (radius - rm)).log()
    return height, azimuth


def transform_to_code(
    fixture: separated.KerrFixture,
    radius_value: float,
    theta_value: float,
    frequency: float,
    mode: int,
    metric: Sequence[BiJet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = BiJet.radial(radius_value)
    theta = BiJet.angular(theta_value)
    cosine = theta.cos()
    sigma = radius * radius + fixture.spin**2 * cosine * cosine
    delta = radius * radius - 2.0 * fixture.mass * radius + fixture.spin**2
    boost = delta / (2.0 * sigma)
    phase = -(radius + 1j * fixture.spin * cosine) / (
        radius - 1j * fixture.spin * cosine
    )
    height, azimuth = tortoise_and_azimuth(fixture, radius)
    modal_phase = (1j * frequency * height - 1j * mode * azimuth).exp()
    transformed = (
        boost * boost * metric[0] * modal_phase,
        boost * phase * metric[1] * modal_phase,
        phase * phase * metric[2] * modal_phase,
    )
    return tuple(r_to_code_R(value, radius_value) for value in transformed)  # type: ignore[return-value]


def r_to_code_R(value: BiJet, radius_value: float) -> np.ndarray:
    """Compose a BL-r Taylor jet with r=1/R, retaining second derivatives."""
    delta_r = np.zeros(RADIAL_DEGREE + 1, dtype=np.complex128)
    for order in range(1, RADIAL_DEGREE + 1):
        delta_r[order] = (-1) ** order * radius_value ** (order + 1)
    result = np.zeros((3, 3), dtype=np.complex128)
    for angular_order in range(3):
        composed = compose_taylor(value.c[:, angular_order], delta_r)
        result[:, angular_order] = composed[:3]
    return result


def compose_taylor(outer: np.ndarray, inner_delta: np.ndarray) -> np.ndarray:
    result = np.zeros_like(inner_delta)
    power = separated.jet_constant(1.0, len(inner_delta) - 1)
    for coefficient in outer:
        result += coefficient * power
        power = separated.jet_multiply(power, inner_delta)
    return result


def check_coordinate_and_tetrad_chain(fixture: separated.KerrFixture) -> None:
    """Independently recover every nonzero code-tetrad component."""
    for radius, theta in POINTS:
        sine = math.sin(theta)
        cosine = math.cos(theta)
        delta = radius**2 - 2.0 * fixture.mass * radius + fixture.spin**2
        sigma = radius**2 + fixture.spin**2 * cosine**2
        boost = delta / (2.0 * sigma)
        height_prime = (
            (radius**2 + fixture.spin**2) / delta
            - 2.0
            - 4.0 * fixture.mass / radius
        )
        azimuth_prime = fixture.spin / delta
        dR_dr = -1.0 / radius**2
        q = -(radius + 1j * fixture.spin * cosine) / (
            radius - 1j * fixture.spin * cosine
        )

        l_kin = np.asarray(
            ((radius**2 + fixture.spin**2) / delta, 1.0, 0.0, fixture.spin / delta),
            dtype=np.complex128,
        )
        n_kin = np.asarray(
            ((radius**2 + fixture.spin**2) / (2.0 * sigma), -delta / (2.0 * sigma), 0.0, fixture.spin / (2.0 * sigma)),
            dtype=np.complex128,
        )
        m_kin = np.asarray(
            (1j * fixture.spin * sine, 0.0, 1.0, 1j / sine),
            dtype=np.complex128,
        ) / (math.sqrt(2.0) * (radius + 1j * fixture.spin * cosine))

        def jacobian(vector: np.ndarray) -> np.ndarray:
            return np.asarray(
                (
                    vector[0] + height_prime * vector[1],
                    dR_dr * vector[1],
                    vector[2],
                    vector[3] + azimuth_prime * vector[1],
                )
            )

        actual_l = boost * jacobian(l_kin)
        actual_n = jacobian(n_kin) / boost
        actual_m = q * jacobian(m_kin)
        code_R = 1.0 / radius
        denominator = 1.0 + fixture.spin**2 * code_R**2 * cosine**2
        expected_l = code_R**2 / denominator * np.asarray(
            (
                2.0 * fixture.mass * (2.0 * fixture.mass - fixture.spin**2 * code_R),
                -0.5 * (1.0 - 2.0 * fixture.mass * code_R + fixture.spin**2 * code_R**2),
                0.0,
                fixture.spin,
            )
        )
        expected_n = np.asarray(
            (2.0 + 4.0 * fixture.mass * code_R, code_R**2, 0.0, 0.0)
        )
        expected_m = code_R / (
            math.sqrt(2.0) * (1.0 - 1j * fixture.spin * code_R * cosine)
        ) * np.asarray((-1j * fixture.spin * sine, 0.0, -1.0, -1j / sine))
        for name, actual, expected in (
            ("l", actual_l, expected_l),
            ("n", actual_n, expected_n),
            ("m", actual_m, expected_m),
        ):
            error = float(np.max(np.abs(actual - expected)))
            if error >= 3.0e-15:
                raise AssertionError(f"{name} tetrad Jacobian mismatch: {error:.3e}")
        if abs(abs(q) - 1.0) >= 2.0e-15:
            raise AssertionError("spin rotation is not unit modulus")
    print("PASS full a!=0 Kinnersley-to-code Jacobian, boost, and spin phase")


def check_bijet_calculus() -> None:
    radius = BiJet.radial(3.2)
    theta = BiJet.angular(0.9)
    function = ((0.3 * radius - 0.7j * theta).exp() / (radius + theta.sin()))
    value = np.exp(0.3 * 3.2 - 0.7j * 0.9) / (3.2 + np.sin(0.9))
    d_radius = value * (0.3 - 1.0 / (3.2 + np.sin(0.9)))
    d_theta = value * (-0.7j - np.cos(0.9) / (3.2 + np.sin(0.9)))
    for name, actual, expected in (
        ("value", function.c[0, 0], value),
        ("radial derivative", function.c[1, 0], d_radius),
        ("angular derivative", function.c[0, 1], d_theta),
    ):
        if abs(actual - expected) >= 3.0e-15:
            raise AssertionError(f"bivariate jet {name} failed: {actual-expected}")
    print("PASS independent bivariate Taylor product quotient and chain rules")


def check_schwarzschild_reconstruction_limit() -> None:
    original = schwarzschild.Fixture()
    partner = schwarzschild.Fixture(omega=-original.omega, m=-original.m)
    radius = 4.0
    theta = 1.1
    integration = {
        "series_order": separated.LEVELS[-1][2],
        "rtol": separated.LEVELS[-1][3],
        "atol": separated.LEVELS[-1][4],
    }
    mode = schwarzschild.normalized_radial_modes(
        partner,
        2,
        (radius,),
        start=-separated.LEVELS[-1][1],
        **integration,
    )[0]
    angular = (1.0 + math.cos(theta)) ** 2
    angular_derivative = -2.0 * math.sin(theta) * (1.0 + math.cos(theta))
    fixture = separated.KerrFixture(
        spin=0.0, omega=-original.omega, m=-original.m, lambda_plus=0.0
    )
    actual = partner_org_h(
        fixture,
        radius,
        theta,
        mode,
        np.asarray((angular, angular_derivative)),
    )
    radial_factors = schwarzschild.partner_h_radial(partner, radius, mode)
    angular_factors = schwarzschild.angular_factors(theta)
    maximum = 0.0
    for actual_component, radial_factor, angular_factor in zip(
        actual, radial_factors, angular_factors
    ):
        expected = np.outer(radial_factor, angular_factor)
        maximum = max(
            maximum,
            float(np.max(np.abs(actual_component.c[:3, :3] - expected))),
        )
    if maximum >= 2.0e-11:
        raise AssertionError(f"Berens Kerr reconstruction misses Schwarzschild limit: {maximum:.3e}")
    print(f"PASS Berens Eq.2.52 Schwarzschild reduction max_error={maximum:.3e}")


def expected_psi0(
    fixture: separated.KerrFixture,
    radius: float,
    theta: float,
    frequency: float,
    mode: int,
    amplitude: complex,
    radial_value: complex,
    angular_value: complex,
) -> complex:
    delta = radius**2 - 2.0 * fixture.mass * radius + fixture.spin**2
    sigma = radius**2 + fixture.spin**2 * math.cos(theta) ** 2
    boost = delta / (2.0 * sigma)
    q = -(radius + 1j * fixture.spin * math.cos(theta)) / (
        radius - 1j * fixture.spin * math.cos(theta)
    )
    rjet = BiJet.radial(radius)
    height, azimuth = tortoise_and_azimuth(fixture, rjet)
    coordinate_phase = np.exp(
        1j * frequency * height.c[0, 0] - 1j * mode * azimuth.c[0, 0]
    )
    return boost**2 * q**2 * amplitude * radial_value * angular_value * coordinate_phase


def extract_component(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radial = np.asarray((value[0, 0], value[1, 0], value[2, 0]))
    angular = np.asarray((value[0, 0], value[0, 1], value[0, 2]))
    return radial, angular


def level_data(
    level: tuple[str, float, int, float, float]
) -> list[dict[str, object]]:
    name, endpoint, series_order, rtol, atol = level
    lambda_plus = separated.solve_angular_eigenvalue(
        2,
        2,
        0.12,
        2,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    original = separated.KerrFixture(lambda_plus=lambda_plus)
    partner = separated.KerrFixture(
        omega=-original.omega, m=-original.m, lambda_plus=lambda_plus
    )
    radii = tuple(point[0] for point in POINTS)
    thetas = tuple(point[1] for point in POINTS)
    original_radial = separated.normalized_radial_modes(
        original,
        2,
        radii,
        start=-endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    partner_radial = separated.normalized_radial_modes(
        partner,
        2,
        radii,
        start=-endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    original_angular = separated.angular_mode(
        original,
        2,
        thetas,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    partner_angular = separated.angular_mode(
        partner,
        2,
        thetas,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    d_product, c_product = separated.starobinsky_products(original)
    omega_plus = original.spin / (2.0 * original.mass * original.r_plus)
    w = 4.0 * original.mass * (original.omega - original.m * omega_plus) * original.r_plus
    gamma = (
        (w + 2j * original.sigma)
        * (w + 1j * original.sigma)
        * w
        * (w - 1j * original.sigma)
    )
    c_prime = c_product / gamma
    b_org = 64.0 / np.conj(c_prime)
    same_amplitude = 4.0 * 24.0 / c_prime
    sharp_amplitude = 48j * original.omega * original.mass / np.conj(c_prime)
    result: list[dict[str, object]] = []
    for point_index, (point, r_plus, r_partner, s_plus, s_partner) in enumerate(
        zip(POINTS, original_radial, partner_radial, original_angular, partner_angular)
    ):
        radius, theta = point
        partner_h = partner_org_h(partner, radius, theta, r_partner, s_partner)
        sharp_metric = tuple(b_org * value for value in partner_h)
        same_metric = (
            np.conj(b_org) * partner_h[0].conj(),
            BiJet.constant(0.0),
            BiJet.constant(0.0),
        )
        same_code = transform_to_code(
            original, radius, theta, original.omega, original.m, same_metric
        )
        sharp_code = transform_to_code(
            partner, radius, theta, partner.omega, partner.m, sharp_metric
        )
        common = {
            "level": name,
            "point": point_index,
            "radius": 1.0 / radius,
            "theta": theta,
        }
        for sector, frequency, mode, metric, expected in (
            (
                "same_mode",
                original.omega,
                original.m,
                same_code,
                expected_psi0(
                    original,
                    radius,
                    theta,
                    original.omega,
                    original.m,
                    same_amplitude,
                    r_plus[0],
                    s_plus[0],
                ),
            ),
            (
                "sharp_partner",
                partner.omega,
                partner.m,
                sharp_code,
                expected_psi0(
                    partner,
                    radius,
                    theta,
                    partner.omega,
                    partner.m,
                    sharp_amplitude,
                    r_partner[0],
                    s_partner[0],
                ),
            ),
        ):
            result.append(
                common
                | {
                    "sector": sector,
                    "omega": frequency,
                    "m": mode,
                    "metric": metric,
                    "expected": expected,
                }
            )
    return result


def maximum_difference(left: list[dict[str, object]], right: list[dict[str, object]]) -> float:
    maximum = 0.0
    for lhs, rhs in zip(left, right):
        for lhs_component, rhs_component in zip(lhs["metric"], rhs["metric"]):
            scale = np.maximum(np.abs(rhs_component), 1.0e-300)
            maximum = max(maximum, float(np.max(np.abs(lhs_component - rhs_component) / scale)))
        maximum = max(
            maximum,
            abs(lhs["expected"] - rhs["expected"]) / max(abs(rhs["expected"]), 1.0e-300),
        )
    return maximum


def c(value: complex) -> str:
    return f"c({value.real:.17e}, {value.imag:.17e})"


def component(value: np.ndarray) -> str:
    radial, angular = extract_component(value)
    return (
        "component({"
        + ", ".join(c(complex(item)) for item in radial)
        + "}, {"
        + ", ".join(c(complex(item)) for item in angular)
        + "}, "
        + c(complex(value[1, 1]))
        + ")"
    )


def render_header(data: list[dict[str, object]]) -> str:
    dependency = SCRIPT.parent / "generate_plus2_tsi_kerr_separated_fixture.py"
    schwarzschild_dependency = (
        SCRIPT.parent / "generate_plus2_tsi_schwarzschild_t0_fixture.py"
    )
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace teuk::test::plus2_tsi_kerr_t0 {",
        "",
        "// Generated file. Do not hand edit.",
        f'inline constexpr const char* generator_sha256 = "{hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* separated_solver_sha256 = "{hashlib.sha256(dependency.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* schwarzschild_generator_sha256 = "{hashlib.sha256(schwarzschild_dependency.read_bytes()).hexdigest()}";',
        f'inline constexpr const char* berens_arxiv_tex_sha256 = "{BERENS_ARXIV_TEX_SHA256}";',
        f'inline constexpr const char* berens_supplement_commit = "{BERENS_SUPPLEMENT_COMMIT}";',
        f'inline constexpr const char* berens_example_nb_sha256 = "{BERENS_EXAMPLE_NB_SHA256}";',
        f'inline constexpr const char* ripley_arxiv_tex_sha256 = "{RIPLEY_ARXIV_TEX_SHA256}";',
        "struct C { double real; double imag; };",
        "struct Component {",
        "  std::array<C, 3> radial;  // value, d_R, d_RR/2",
        "  std::array<C, 3> angular; // value, d_theta, d_thetatheta/2",
        "  C mixed_Rtheta;",
        "};",
        "struct Fixture {",
        "  const char* level; const char* sector; int point;",
        "  double radius, theta, omega; int m;",
        "  std::array<Component, 3> metric; C expected_psi0_code;",
        "};",
        "constexpr C c(double real, double imag) { return {real, imag}; }",
        "constexpr Component component(const std::array<C, 3>& radial,",
        "                              const std::array<C, 3>& angular,",
        "                              C mixed_Rtheta) {",
        "  return {radial, angular, mixed_Rtheta};",
        "}",
        f"inline std::array<Fixture, {len(data)}> fixtures() {{",
        "  return {",
    ]
    for item in data:
        components = ",\n        ".join(component(value) for value in item["metric"])
        lines.extend(
            [
                "    Fixture{",
                f'      "{item["level"]}", "{item["sector"]}", {item["point"]},',
                f'      {item["radius"]:.17e}, {item["theta"]:.17e}, {item["omega"]:.17e}, {item["m"]},',
                "      std::array<Component, 3>{",
                f"        {components}",
                "      },",
                f'      {c(complex(item["expected"]))}',
                "    },",
            ]
        )
    lines.extend(["  };", "}", "", "}  // namespace teuk::test::plus2_tsi_kerr_t0", ""])
    return "\n".join(lines)


def run_fixture() -> list[dict[str, object]]:
    check_bijet_calculus()
    check_schwarzschild_reconstruction_limit()
    check_coordinate_and_tetrad_chain(separated.KerrFixture())
    levels = [level_data(level) for level in separated.LEVELS]
    coarse_medium = maximum_difference(levels[0], levels[1])
    medium_fine = maximum_difference(levels[1], levels[2])
    print(
        f"Kerr T0 fixture convergence coarse-medium={coarse_medium:.6e} "
        f"medium-fine={medium_fine:.6e}"
    )
    if not medium_fine < coarse_medium or medium_fine >= 5.0e-9:
        raise AssertionError("Kerr T0 fixture failed normalized-mode convergence gate")
    print("PASS normalized moderate-Kerr ORG metric/T0 fixture generation")
    return [item for level in levels for item in level]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    rendered = render_header(run_fixture())
    if args.output:
        args.output.write_text(rendered)
    elif args.check:
        if args.check.read_text() != rendered:
            raise AssertionError(f"generated fixture differs from {args.check}")
        print(f"PASS generated Kerr T0 fixture matches {args.check}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
