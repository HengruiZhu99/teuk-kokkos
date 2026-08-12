#!/usr/bin/env python3
"""Generate an independently normalized moderate-Kerr separated TSI fixture.

This script implements Berens--Gravely--Lupsasca arXiv:2403.20311
Eqs. (2.6)--(2.8), (2.17)--(2.18), (2.25), (2.28)--(2.29), and
(2.37)--(2.38).  The angular eigenvalue is obtained from the class-I
confluent-Heun Wronskian in their Eq. (B.21), not from production code.

The fixture deliberately uses a real frequency.  It establishes the hatted
normalization and both first-form separated identities at a/M=0.6, but it is
not a QNM or a field-level T0[h] fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

SCRIPT = Path(__file__).resolve()

BERENS_ARXIV_TEX_SHA256 = (
    "f5af07f8692e4f8d49f271e41056eadb0bf180e78a6c8574f238930f2298b3cf"
)
BERENS_SUPPLEMENT_COMMIT = "cf924707593a58ec889c70ea501d764e99d1d4aa"
BERENS_EXAMPLE_NB_SHA256 = (
    "d2c69c27f34c4da852c355a3240a1f351c05340182ee740bdf8fa56e5c05c974"
)

LEVELS = (
    ("coarse", 1.0e-3, 28, 2.0e-10, 2.0e-12),
    ("medium", 5.0e-4, 36, 2.0e-12, 2.0e-14),
    ("fine", 2.5e-4, 44, 2.0e-13, 2.0e-15),
)
RADII = (2.3, 3.0, 4.5, 8.0)
THETAS = (0.7, 1.1, 1.8, 2.4)


@dataclass(frozen=True)
class HeunParameters:
    q: complex
    alpha: complex
    gamma: complex
    delta: complex
    epsilon: complex


@dataclass(frozen=True)
class KerrFixture:
    mass: float = 1.0
    spin: float = 0.6
    omega: float = 0.2
    ell: int = 2
    m: int = 2
    lambda_plus: float = 0.0

    @property
    def r_plus(self) -> float:
        return self.mass + math.sqrt(self.mass**2 - self.spin**2)

    @property
    def r_minus(self) -> float:
        return self.mass - math.sqrt(self.mass**2 - self.spin**2)

    @property
    def sigma(self) -> float:
        return self.r_plus - self.r_minus

    @property
    def c(self) -> float:
        return self.spin * self.omega

    def separation_constant(self, spin_weight: int) -> float:
        if spin_weight == 2:
            return self.lambda_plus
        if spin_weight == -2:
            return self.lambda_plus + 4.0
        raise ValueError("fixture supports only s=+/-2")


def heun_series(parameters: HeunParameters, order: int) -> np.ndarray:
    """Canonical H(0)=1 coefficients from Berens Eq. (A.2a)."""
    coefficients = np.zeros(order + 1, dtype=np.complex128)
    coefficients[0] = 1.0
    for k in range(order):
        previous = coefficients[k - 1] if k else 0j
        numerator = (
            k
            * (k - 1)
            + k * (parameters.gamma + parameters.delta - parameters.epsilon)
            - parameters.q
        ) * coefficients[k] + (
            parameters.epsilon * (k - 1) + parameters.alpha
        ) * previous
        denominator = (k + 1) * (k + parameters.gamma)
        if abs(denominator) < 1.0e-14:
            raise ArithmeticError("resonant Heun recurrence is unsupported")
        coefficients[k + 1] = numerator / denominator
    return coefficients


def evaluate_series(coefficients: np.ndarray, z: float) -> tuple[complex, complex]:
    value = 0j
    derivative = 0j
    for n in range(len(coefficients) - 1, 0, -1):
        derivative = derivative * z + n * coefficients[n]
        value = value * z + coefficients[n]
    return value * z + coefficients[0], derivative


def integrate_heun(
    parameters: HeunParameters,
    start: float,
    target: Sequence[float],
    *,
    series_order: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    coefficients = heun_series(parameters, series_order)
    initial = evaluate_series(coefficients, start)

    def rhs(z: float, value: np.ndarray) -> np.ndarray:
        h, dh = value
        d2h = -(
            parameters.gamma / z
            + parameters.delta / (z - 1.0)
            + parameters.epsilon
        ) * dh - (
            (parameters.alpha * z - parameters.q) / (z * (z - 1.0))
        ) * h
        return np.asarray((dh, d2h), dtype=np.complex128)

    increasing = target[-1] > start
    ordered = sorted(target, reverse=not increasing)
    solution = solve_ivp(
        rhs,
        (start, ordered[-1]),
        np.asarray(initial, dtype=np.complex128),
        method="DOP853",
        t_eval=np.asarray(ordered),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Heun integration failed: {solution.message}")
    by_point = {float(point): solution.y[:, i] for i, point in enumerate(solution.t)}
    return np.asarray([by_point[float(point)] for point in target])


def angular_parameters(
    spin_weight: int, m: int, c: float, separation_constant: float
) -> HeunParameters:
    """Berens Eqs. (2.6)--(2.8), with the z_+ class-I choice."""
    mu1 = abs(spin_weight + m) / 2.0
    mu2 = abs(spin_weight - m) / 2.0
    beta = 2.0 * c * (mu1 + mu2 + spin_weight + 1.0)
    p = (
        -separation_constant
        - spin_weight * (spin_weight + 1.0)
        + 2.0 * c * (mu1 - mu2 - m)
        + (mu1 + mu2) ** 2
        + mu1
        + mu2
    )
    return HeunParameters(
        q=-p + beta,
        alpha=2.0 * beta,
        gamma=2.0 * mu2 + 1.0,
        delta=2.0 * mu1 + 1.0,
        epsilon=4.0 * c,
    )


def reflected_heun(parameters: HeunParameters) -> HeunParameters:
    """Class-I solution regular at z=1, Berens Eq. (B.21)."""
    return HeunParameters(
        q=parameters.q - parameters.alpha,
        alpha=-parameters.alpha,
        gamma=parameters.delta,
        delta=parameters.gamma,
        epsilon=-parameters.epsilon,
    )


def angular_wronskian(
    spin_weight: int,
    m: int,
    c: float,
    separation_constant: float,
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> float:
    parameters = angular_parameters(spin_weight, m, c, separation_constant)
    left = integrate_heun(
        parameters,
        endpoint,
        (0.5,),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )[0]
    right_w = integrate_heun(
        reflected_heun(parameters),
        endpoint,
        (0.5,),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )[0]
    # The reflected solution uses w=1-z, hence d/dz=-d/dw.
    wronskian = left[0] * (-right_w[1]) - right_w[0] * left[1]
    if abs(wronskian.imag) > 2.0e-10:
        raise AssertionError(f"real-frequency angular Wronskian became complex: {wronskian}")
    return float(wronskian.real)


def solve_angular_eigenvalue(
    spin_weight: int,
    m: int,
    c: float,
    ell: int,
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> float:
    spherical = ell * (ell + 1) - spin_weight * (spin_weight + 1)

    def residual(value: float) -> float:
        return angular_wronskian(
            spin_weight,
            m,
            c,
            value,
            endpoint=endpoint,
            series_order=series_order,
            rtol=rtol,
            atol=atol,
        )

    grid = np.linspace(spherical - 2.0, spherical + 2.0, 81)
    values = [residual(float(value)) for value in grid]
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left * f_right < 0.0:
            roots.append(brentq(residual, float(left), float(right), xtol=5.0e-14))
    if not roots:
        raise AssertionError("failed to bracket the angular eigenvalue")
    return min(roots, key=lambda value: abs(value - spherical))


def angular_mode(
    fixture: KerrFixture,
    spin_weight: int,
    thetas: Sequence[float],
    *,
    endpoint: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Return hatted S and dS/dtheta with H(0)=1."""
    parameters = angular_parameters(
        spin_weight,
        fixture.m,
        fixture.c,
        fixture.separation_constant(spin_weight),
    )
    z_values = [(1.0 + math.cos(theta)) / 2.0 for theta in thetas]
    heun = integrate_heun(
        parameters,
        endpoint,
        tuple(sorted(z_values)),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    by_z = {float(z): value for z, value in zip(sorted(z_values), heun)}
    result = np.empty((len(thetas), 2), dtype=np.complex128)
    mu1 = abs(spin_weight + fixture.m) / 2.0
    mu2 = abs(spin_weight - fixture.m) / 2.0
    for index, (theta, z) in enumerate(zip(thetas, z_values)):
        h, dh_dz = by_z[float(z)]
        u = math.cos(theta)
        prefactor = (1.0 - u) ** mu1 * (1.0 + u) ** mu2 * math.exp(
            fixture.c * (1.0 + u)
        )
        dlog_du = -mu1 / (1.0 - u) + mu2 / (1.0 + u) + fixture.c
        result[index, 0] = prefactor * h
        result[index, 1] = -math.sin(theta) * prefactor * (dlog_du * h + 0.5 * dh_dz)
    return result


def jet_constant(value: complex, degree: int) -> np.ndarray:
    result = np.zeros(degree + 1, dtype=np.complex128)
    result[0] = value
    return result


def jet_variable(value: float, degree: int) -> np.ndarray:
    result = jet_constant(value, degree)
    if degree:
        result[1] = 1.0
    return result


def jet_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    degree = min(len(left), len(right)) - 1
    return np.convolve(left, right)[: degree + 1]


def jet_inverse(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    result[0] = 1.0 / value[0]
    for n in range(1, len(value)):
        result[n] = -sum(value[k] * result[n - k] for k in range(1, n + 1)) / value[0]
    return result


def jet_divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return jet_multiply(left, jet_inverse(right))


def jet_derivative(value: np.ndarray) -> np.ndarray:
    return np.asarray([(n + 1) * value[n + 1] for n in range(len(value) - 1)])


def exp_jet(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    result[0] = np.exp(value[0])
    for n in range(1, len(value)):
        result[n] = sum(k * value[k] * result[n - k] for k in range(1, n + 1)) / n
    return result


def sin_jet(value: np.ndarray) -> np.ndarray:
    return (exp_jet(1j * value) - exp_jet(-1j * value)) / (2j)


def cos_jet(value: np.ndarray) -> np.ndarray:
    return 0.5 * (exp_jet(1j * value) + exp_jet(-1j * value))


def on_shell_angular_jet(
    fixture: KerrFixture,
    spin_weight: int,
    theta: float,
    value: complex,
    derivative: complex,
    degree: int,
) -> np.ndarray:
    coordinate = jet_variable(theta, degree)
    sine = sin_jet(coordinate)
    cosine = cos_jet(coordinate)
    one = jet_constant(1.0, degree)
    cotangent = jet_divide(cosine, sine)
    a_coefficient = -cotangent
    angular_a = (
        fixture.separation_constant(spin_weight)
        + 2.0 * fixture.spin * fixture.m * fixture.omega
        - fixture.c**2
    )
    potential = (
        fixture.c**2 * jet_multiply(cosine, cosine)
        - jet_divide(
            jet_multiply(
                fixture.m * one + spin_weight * cosine,
                fixture.m * one + spin_weight * cosine,
            ),
            jet_multiply(sine, sine),
        )
        - 2.0 * spin_weight * fixture.c * cosine
        + (spin_weight + angular_a) * one
    )
    b_coefficient = -potential
    result = np.zeros(degree + 1, dtype=np.complex128)
    result[0] = value
    result[1] = derivative
    for n in range(degree - 1):
        rhs = 0j
        for k in range(n + 1):
            rhs += (
                a_coefficient[k] * (n - k + 1) * result[n - k + 1]
                + b_coefficient[k] * result[n - k]
            )
        result[n + 2] = rhs / ((n + 2) * (n + 1))
    return result


def angular_operator(
    fixture: KerrFixture,
    theta: float,
    value: np.ndarray,
    n: int,
    *,
    dagger: bool,
) -> np.ndarray:
    coordinate = jet_variable(theta, len(value) - 1)
    sine = sin_jet(coordinate)
    cosine = cos_jet(coordinate)
    q = -fixture.c * sine + fixture.m * jet_inverse(sine)
    algebraic = (-q if dagger else q) + n * jet_divide(cosine, sine)
    return jet_derivative(value) + jet_multiply(algebraic, value)[:-1]


def starobinsky_products(fixture: KerrFixture) -> tuple[float, float]:
    x = fixture.lambda_plus
    c = fixture.c
    d_product = (
        (x + 4.0) ** 2 * (x + 6.0) ** 2
        + 8.0 * c * (fixture.m - c) * (x + 4.0) * (5.0 * x + 26.0)
        + 48.0 * c**2 * (2.0 * x + 8.0 + 3.0 * (fixture.m - c) ** 2)
    )
    return d_product, d_product + (12.0 * fixture.mass * fixture.omega) ** 2


def angular_residuals(
    fixture: KerrFixture, plus: np.ndarray, minus: np.ndarray
) -> tuple[float, float]:
    d_product, _ = starobinsky_products(fixture)
    d_hat = d_product / 24.0
    d_hat_prime = 24.0
    forward = []
    backward = []
    for theta, plus_value, minus_value in zip(THETAS, plus, minus):
        plus_jet = on_shell_angular_jet(
            fixture, 2, theta, plus_value[0], plus_value[1], 4
        )
        minus_jet = on_shell_angular_jet(
            fixture, -2, theta, minus_value[0], minus_value[1], 4
        )
        for n in (2, 1, 0, -1):
            plus_jet = angular_operator(fixture, theta, plus_jet, n, dagger=False)
            minus_jet = angular_operator(fixture, theta, minus_jet, n, dagger=True)
        forward.append(abs(plus_jet[0] - d_hat * minus_value[0]) / abs(d_hat * minus_value[0]))
        backward.append(
            abs(minus_jet[0] - d_hat_prime * plus_value[0])
            / abs(d_hat_prime * plus_value[0])
        )
    return max(forward), max(backward)


def radial_heun_parameters(fixture: KerrFixture, spin_weight: int) -> HeunParameters:
    rp = fixture.r_plus
    rm = fixture.r_minus
    sigma = fixture.sigma
    omega_plus = fixture.spin / (2.0 * fixture.mass * rp)
    omega_minus = fixture.spin / (2.0 * fixture.mass * rm)
    xi1 = 1j * (2.0 * fixture.mass * rp / sigma) * (
        fixture.omega - fixture.m * omega_plus
    )
    xi2 = -1j * (2.0 * fixture.mass * rm / sigma) * (
        fixture.omega - fixture.m * omega_minus
    )
    gamma = 2.0 * xi1 + spin_weight + 1.0
    delta = 2.0 * xi2 + spin_weight + 1.0
    epsilon = -2j * fixture.omega * sigma
    alpha = -2j * fixture.omega * (2 * spin_weight + 1) * sigma
    q = -2j * fixture.omega * rp * (2 * spin_weight + 1) + fixture.separation_constant(
        spin_weight
    )
    return HeunParameters(
        q=q + (epsilon - delta) * (1.0 - gamma),
        alpha=alpha + epsilon * (1.0 - gamma),
        gamma=2.0 - gamma,
        delta=delta,
        epsilon=epsilon,
    )


def normalized_radial_modes(
    fixture: KerrFixture,
    spin_weight: int,
    radii: Sequence[float],
    *,
    start: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    parameters = radial_heun_parameters(fixture, spin_weight)
    target_z = [-(radius - fixture.r_plus) / fixture.sigma for radius in radii]
    values = integrate_heun(
        parameters,
        start,
        tuple(target_z),
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    result = np.empty((len(radii), 2), dtype=np.complex128)
    xi1 = (1.0 - parameters.gamma - spin_weight) / 2.0
    xi2 = (parameters.delta - spin_weight - 1.0) / 2.0
    for index, (radius, (h, dh_dz)) in enumerate(zip(radii, values)):
        x = radius - fixture.r_plus
        prefactor = (
            fixture.sigma ** (-xi2 - spin_weight)
            * x ** (-xi1 - spin_weight)
            * (radius - fixture.r_minus) ** xi2
            * np.exp(1j * fixture.omega * x)
        )
        dlog = (
            (-xi1 - spin_weight) / x
            + xi2 / (radius - fixture.r_minus)
            + 1j * fixture.omega
        )
        result[index] = (prefactor * h, prefactor * (dlog * h - dh_dz / fixture.sigma))
    return result


def on_shell_radial_jet(
    fixture: KerrFixture,
    spin_weight: int,
    radius: float,
    value: complex,
    derivative: complex,
    degree: int,
) -> np.ndarray:
    r = jet_variable(radius, degree)
    one = jet_constant(1.0, degree)
    delta = jet_multiply(r, r) - 2.0 * fixture.mass * r + fixture.spin**2 * one
    delta_prime = 2.0 * (r - fixture.mass * one)
    k = fixture.omega * (jet_multiply(r, r) + fixture.spin**2 * one) - fixture.spin * fixture.m * one
    a_coefficient = -(spin_weight + 1.0) * jet_divide(delta_prime, delta)
    potential_over_delta = jet_divide(
        jet_multiply(k, k) - 2j * spin_weight * jet_multiply(r - fixture.mass * one, k),
        jet_multiply(delta, delta),
    ) + jet_divide(
        4j * spin_weight * fixture.omega * r
        - fixture.separation_constant(spin_weight) * one,
        delta,
    )
    b_coefficient = -potential_over_delta
    result = np.zeros(degree + 1, dtype=np.complex128)
    result[0] = value
    result[1] = derivative
    for n in range(degree - 1):
        rhs = 0j
        for k_index in range(n + 1):
            rhs += (
                a_coefficient[k_index]
                * (n - k_index + 1)
                * result[n - k_index + 1]
                + b_coefficient[k_index] * result[n - k_index]
            )
        result[n + 2] = rhs / ((n + 2) * (n + 1))
    return result


def radial_operator(
    fixture: KerrFixture, radius: float, value: np.ndarray, *, dagger: bool
) -> np.ndarray:
    degree = len(value) - 1
    r = jet_variable(radius, degree)
    one = jet_constant(1.0, degree)
    delta = jet_multiply(r, r) - 2.0 * fixture.mass * r + fixture.spin**2 * one
    k = fixture.omega * (jet_multiply(r, r) + fixture.spin**2 * one) - fixture.spin * fixture.m * one
    signed = (1j if dagger else -1j) * jet_divide(k, delta)
    return jet_derivative(value) + jet_multiply(signed, value)[:-1]


def delta_squared(fixture: KerrFixture, radius: float, value: np.ndarray) -> np.ndarray:
    degree = len(value) - 1
    r = jet_variable(radius, degree)
    one = jet_constant(1.0, degree)
    delta = jet_multiply(r, r) - 2.0 * fixture.mass * r + fixture.spin**2 * one
    return jet_multiply(jet_multiply(delta, delta), value)


def repeat_radial_operator(
    fixture: KerrFixture, radius: float, value: np.ndarray, *, dagger: bool
) -> np.ndarray:
    for _ in range(4):
        value = radial_operator(fixture, radius, value, dagger=dagger)
    return value


def radial_residuals(
    fixture: KerrFixture, plus: np.ndarray, minus: np.ndarray
) -> tuple[float, float, float, float]:
    d_product, c_product = starobinsky_products(fixture)
    omega_plus = fixture.spin / (2.0 * fixture.mass * fixture.r_plus)
    k_horizon = fixture.omega - fixture.m * omega_plus
    w = 4.0 * fixture.mass * k_horizon * fixture.r_plus
    gamma = (
        (w + 2j * fixture.sigma)
        * (w + 1j * fixture.sigma)
        * w
        * (w - 1j * fixture.sigma)
    )
    gamma_prime = c_product / gamma
    residuals = [[], [], [], []]
    for radius, plus_value, minus_value in zip(RADII, plus, minus):
        plus_jet = on_shell_radial_jet(
            fixture, 2, radius, plus_value[0], plus_value[1], 8
        )
        minus_jet = on_shell_radial_jet(
            fixture, -2, radius, minus_value[0], minus_value[1], 8
        )
        minus_to_plus = repeat_radial_operator(
            fixture, radius, minus_jet[:5], dagger=False
        )[0]
        plus_to_minus = delta_squared(
            fixture,
            radius,
            repeat_radial_operator(
                fixture, radius, delta_squared(fixture, radius, plus_jet[:5]), dagger=True
            ),
        )[0]
        second_plus = repeat_radial_operator(
            fixture,
            radius,
            delta_squared(
                fixture,
                radius,
                repeat_radial_operator(
                    fixture, radius, delta_squared(fixture, radius, plus_jet), dagger=True
                ),
            ),
            dagger=False,
        )[0]
        second_minus = delta_squared(
            fixture,
            radius,
            repeat_radial_operator(
                fixture,
                radius,
                delta_squared(
                    fixture,
                    radius,
                    repeat_radial_operator(fixture, radius, minus_jet, dagger=False),
                ),
                dagger=True,
            ),
        )[0]
        expected = (
            gamma * plus_value[0],
            gamma_prime * minus_value[0],
            c_product * plus_value[0],
            c_product * minus_value[0],
        )
        for bucket, actual, target in zip(
            residuals,
            (minus_to_plus, plus_to_minus, second_plus, second_minus),
            expected,
        ):
            bucket.append(abs(actual - target) / max(abs(target), 1.0e-300))
    return tuple(max(values) for values in residuals)  # type: ignore[return-value]


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0e-300)))


def evaluate_level(
    name: str, endpoint: float, series_order: int, rtol: float, atol: float
) -> dict[str, object]:
    lambda_plus = solve_angular_eigenvalue(
        2,
        2,
        0.12,
        2,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    lambda_minus = solve_angular_eigenvalue(
        -2,
        2,
        0.12,
        2,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    fixture = KerrFixture(lambda_plus=lambda_plus)
    plus_angular = angular_mode(
        fixture,
        2,
        THETAS,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    minus_angular = angular_mode(
        fixture,
        -2,
        THETAS,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    angular_forward, angular_backward = angular_residuals(
        fixture, plus_angular, minus_angular
    )
    plus_radial = normalized_radial_modes(
        fixture,
        2,
        RADII,
        start=-endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    minus_radial = normalized_radial_modes(
        fixture,
        -2,
        RADII,
        start=-endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    radial = radial_residuals(fixture, plus_radial, minus_radial)

    partner = KerrFixture(omega=-fixture.omega, m=-fixture.m, lambda_plus=lambda_plus)
    partner_plus_radial = normalized_radial_modes(
        partner,
        2,
        RADII,
        start=-endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    partner_plus_angular = angular_mode(
        partner,
        2,
        THETAS,
        endpoint=endpoint,
        series_order=series_order,
        rtol=rtol,
        atol=atol,
    )
    signed_radial = relative_difference(partner_plus_radial, np.conj(plus_radial))
    signed_angular = relative_difference(partner_plus_angular, np.conj(minus_angular))
    d_product, c_product = starobinsky_products(fixture)
    omega_plus = fixture.spin / (2.0 * fixture.mass * fixture.r_plus)
    w = 4.0 * fixture.mass * (fixture.omega - fixture.m * omega_plus) * fixture.r_plus
    gamma = (
        (w + 2j * fixture.sigma)
        * (w + 1j * fixture.sigma)
        * w
        * (w - 1j * fixture.sigma)
    )
    gamma_prime = c_product / gamma
    same_amplitude = 4.0 * 24.0 / gamma_prime
    sharp_amplitude = 48j * fixture.omega * fixture.mass / np.conj(gamma_prime)
    return {
        "name": name,
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "angular_forward": angular_forward,
        "angular_backward": angular_backward,
        "radial": radial,
        "signed_radial": signed_radial,
        "signed_angular": signed_angular,
        "d_product": d_product,
        "c_product": c_product,
        "d_hat": d_product / 24.0,
        "d_hat_prime": 24.0,
        "c_hat": gamma,
        "c_hat_prime": gamma_prime,
        "sharp_to_same": sharp_amplitude / same_amplitude,
        "plus_angular": plus_angular,
        "minus_angular": minus_angular,
        "plus_radial": plus_radial,
        "minus_radial": minus_radial,
    }


def cxx_complex(value: complex) -> str:
    return f"C{{{value.real:.17e}, {value.imag:.17e}}}"


def render_header(levels: Sequence[dict[str, object]]) -> str:
    script_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace teuk::test::plus2_tsi_kerr {",
        "",
        "// Generated file. Do not hand edit.",
        f'inline constexpr const char* generator_sha256 = "{script_hash}";',
        f'inline constexpr const char* berens_arxiv_tex_sha256 = "{BERENS_ARXIV_TEX_SHA256}";',
        f'inline constexpr const char* berens_supplement_commit = "{BERENS_SUPPLEMENT_COMMIT}";',
        f'inline constexpr const char* berens_example_nb_sha256 = "{BERENS_EXAMPLE_NB_SHA256}";',
        "inline constexpr double mass = 1.0;",
        "inline constexpr double spin = 0.6;",
        "inline constexpr double omega = 0.2;",
        "inline constexpr int ell = 2;",
        "inline constexpr int m = 2;",
        "struct C { double real; double imag; };",
        "struct Level {",
        "  const char* name;",
        "  double lambda_plus, lambda_minus;",
        "  double angular_forward, angular_backward;",
        "  std::array<double, 4> radial_residuals;",
        "  double angular_mode_change, radial_mode_change;",
        "  double signed_radial, signed_angular;",
        "  double d_product, c_product, d_hat, d_hat_prime;",
        "  C c_hat, c_hat_prime, sharp_to_same;",
        "};",
        f"inline constexpr std::array<Level, {len(levels)}> levels{{{{",
    ]
    for level in levels:
        radial = ", ".join(f"{value:.17e}" for value in level["radial"])
        lines.extend(
            [
                "  Level{",
                f'    "{level["name"]}",',
                f'    {level["lambda_plus"]:.17e}, {level["lambda_minus"]:.17e},',
                f'    {level["angular_forward"]:.17e}, {level["angular_backward"]:.17e},',
                f"    {{{radial}}},",
                f'    {level["angular_mode_change"]:.17e}, {level["radial_mode_change"]:.17e},',
                f'    {level["signed_radial"]:.17e}, {level["signed_angular"]:.17e},',
                f'    {level["d_product"]:.17e}, {level["c_product"]:.17e},',
                f'    {level["d_hat"]:.17e}, {level["d_hat_prime"]:.17e},',
                f'    {cxx_complex(complex(level["c_hat"]))},',
                f'    {cxx_complex(complex(level["c_hat_prime"]))},',
                f'    {cxx_complex(complex(level["sharp_to_same"]))}',
                "  },",
            ]
        )
    lines.extend(["}};", "", "}  // namespace teuk::test::plus2_tsi_kerr", ""])
    return "\n".join(lines)


def run_fixture() -> list[dict[str, object]]:
    levels = [evaluate_level(*level) for level in LEVELS]
    for index, level in enumerate(levels):
        if index == 0:
            level["angular_mode_change"] = -1.0
            level["radial_mode_change"] = -1.0
            continue
        previous = levels[index - 1]
        level["angular_mode_change"] = max(
            relative_difference(previous["plus_angular"], level["plus_angular"]),
            relative_difference(previous["minus_angular"], level["minus_angular"]),
        )
        level["radial_mode_change"] = max(
            relative_difference(previous["plus_radial"], level["plus_radial"]),
            relative_difference(previous["minus_radial"], level["minus_radial"]),
        )
    for level in levels:
        print(
            f'{level["name"]:6s} lambda+={level["lambda_plus"]:.15g} '
            f'lambda-={level["lambda_minus"]:.15g} '
            f'max_ATSI={max(level["angular_forward"], level["angular_backward"]):.3e} '
            f'max_RTSI={max(level["radial"]):.3e} '
            f'angular_change={level["angular_mode_change"]:.3e} '
            f'radial_change={level["radial_mode_change"]:.3e}'
        )
    eigenvalue_changes = (
        abs(levels[0]["lambda_plus"] - levels[1]["lambda_plus"]),
        abs(levels[1]["lambda_plus"] - levels[2]["lambda_plus"]),
    )
    if not eigenvalue_changes[1] < eigenvalue_changes[0]:
        raise AssertionError(f"angular eigenvalue did not converge: {eigenvalue_changes}")
    if not levels[2]["angular_mode_change"] < levels[1]["angular_mode_change"]:
        raise AssertionError("normalized angular modes did not converge")
    if not levels[2]["radial_mode_change"] < levels[1]["radial_mode_change"]:
        raise AssertionError("normalized radial modes did not converge")
    fine = levels[-1]
    if abs(fine["lambda_minus"] - fine["lambda_plus"] - 4.0) >= 2.0e-10:
        raise AssertionError("independently solved spin eigenvalues violate lambda_- = lambda_+ + 4")
    if max(fine["angular_forward"], fine["angular_backward"]) >= 2.0e-8:
        raise AssertionError("fine angular first-form TSI residual exceeds gate")
    if max(fine["radial"]) >= 5.0e-7:
        raise AssertionError("fine radial TSI residual exceeds gate")
    if max(fine["signed_radial"], fine["signed_angular"]) >= 2.0e-8:
        raise AssertionError("signed real-frequency normalization symmetry failed")
    if abs(fine["d_hat"] * fine["d_hat_prime"] - fine["d_product"]) >= 2.0e-11:
        raise AssertionError("hatted angular constants do not multiply to D")
    c_hat = complex(fine["c_hat"])
    c_hat_prime = complex(fine["c_hat_prime"])
    if abs(c_hat * c_hat_prime - fine["c_product"]) >= 2.0e-11:
        raise AssertionError("hatted radial constants do not multiply to C")
    expected_ratio = 0.5j * KerrFixture().omega * c_hat_prime / np.conj(c_hat_prime)
    if abs(fine["sharp_to_same"] - expected_ratio) >= 2.0e-15:
        raise AssertionError("Eq. (2.44) sharp-sector phase lost a conjugation")
    if abs(fine["sharp_to_same"] - 0.5j * KerrFixture().omega) < 1.0e-3:
        raise AssertionError("Eq. (2.44) phase was reduced to the phase-free shortcut")
    print("PASS normalized moderate-Kerr separated TSI fixture")
    return levels


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
        print(f"PASS generated Kerr fixture matches {args.check}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
