#!/usr/bin/env python3
"""Numerical normalized Schwarzschild radial TSI fixture.

This script implements the horizon-in modes of Berens, Gravely, and
Lupsasca, arXiv:2403.20311, Eqs. (2.17), (2.18), and (2.38), directly from
the canonical confluent-Heun ODE in their Eq. (A.1).  It does not call a
confluent-Heun special-function implementation and does not use production
Teukolsky code.

The fourth- and eighth-order radial operators are evaluated with local Taylor
jets.  Their derivatives are generated independently from the second-order
radial Teukolsky ODE, rather than by fitting sampled mode values.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

Complex = complex


@dataclass(frozen=True)
class Fixture:
    mass: float = 1.0
    spin: float = 0.0
    omega: float = 0.2
    ell: int = 2
    m: int = 2

    @property
    def r_plus(self) -> float:
        return self.mass + math.sqrt(self.mass**2 - self.spin**2)

    @property
    def r_minus(self) -> float:
        return self.mass - math.sqrt(self.mass**2 - self.spin**2)

    @property
    def sigma(self) -> float:
        return self.r_plus - self.r_minus

    def separation_constant(self, spin_weight: int) -> float:
        if self.spin != 0.0:
            raise ValueError("this fixture deliberately supports Schwarzschild only")
        return self.ell * (self.ell + 1) - spin_weight * (spin_weight + 1)


@dataclass(frozen=True)
class HeunParameters:
    q: Complex
    alpha: Complex
    gamma: Complex
    delta: Complex
    epsilon: Complex
    xi1: Complex
    xi2: Complex


def horizon_in_heun_parameters(fixture: Fixture, spin_weight: int) -> HeunParameters:
    """Return Eq. (2.18) parameters after the Eq. (2.17a) in-mode shift."""
    rp = fixture.r_plus
    rm = fixture.r_minus
    sigma = fixture.sigma
    mass = fixture.mass
    a = fixture.spin
    omega = fixture.omega
    m = fixture.m
    c_plus = 2.0 * mass * rp / sigma
    c_minus = 2.0 * mass * rm / sigma
    omega_plus = a / (2.0 * mass * rp)
    # At Schwarzschild r_-=0 and c_-=0.  Avoid the undefined 0/0 form of
    # Omega_- by using the finite product c_- (omega-m Omega_-)=0.
    xi1 = 1j * c_plus * (omega - m * omega_plus)
    xi2 = 0j if c_minus == 0.0 else -1j * c_minus * (omega - m * a / (2.0 * mass * rm))
    gamma = 2.0 * xi1 + spin_weight + 1.0
    delta = 2.0 * xi2 + spin_weight + 1.0
    epsilon = -2j * omega * sigma
    alpha = -2j * omega * (2 * spin_weight + 1) * sigma
    q = -2j * omega * rp * (2 * spin_weight + 1) + fixture.separation_constant(
        spin_weight
    )
    # Eq. (2.17a): H_in is the second local Frobenius solution with the
    # explicit power z^(1-gamma) already carried by the radial prefactor.
    return HeunParameters(
        q=q + (epsilon - delta) * (1.0 - gamma),
        alpha=alpha + epsilon * (1.0 - gamma),
        gamma=2.0 - gamma,
        delta=delta,
        epsilon=epsilon,
        xi1=xi1,
        xi2=xi2,
    )


def heun_series(parameters: HeunParameters, order: int) -> np.ndarray:
    """Canonical HeunC coefficients normalized by a_0=1 (paper Eq. A.2a)."""
    if order < 2:
        raise ValueError("series order must be at least two")
    a = np.zeros(order + 1, dtype=np.complex128)
    a[0] = 1.0
    for k in range(order):
        previous = a[k - 1] if k else 0j
        numerator = (
            k * (k - 1)
            + k * (parameters.gamma + parameters.delta - parameters.epsilon)
            - parameters.q
        ) * a[k] + (parameters.epsilon * (k - 1) + parameters.alpha) * previous
        denominator = (k + 1) * (k + parameters.gamma)
        if abs(denominator) < 1.0e-14:
            raise ArithmeticError("resonant Heun recurrence is unsupported")
        a[k + 1] = numerator / denominator
    return a


def evaluate_series(coefficients: np.ndarray, z: float) -> tuple[Complex, Complex]:
    value = 0j
    derivative = 0j
    for n in range(len(coefficients) - 1, 0, -1):
        derivative = derivative * z + n * coefficients[n]
        value = value * z + coefficients[n]
    value = value * z + coefficients[0]
    return value, derivative


def integrate_heun(
    parameters: HeunParameters,
    target_z: Sequence[float],
    *,
    start: float,
    series_order: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Integrate the normalized local Heun solution from a Frobenius seed."""
    if not (start < 0.0 and all(z < start for z in target_z)):
        raise ValueError("target z values must lie below the negative start point")
    coefficients = heun_series(parameters, series_order)
    h0, dh0 = evaluate_series(coefficients, start)

    def rhs(z: float, y: np.ndarray) -> np.ndarray:
        h, dh = y
        d2h = (
            -(parameters.gamma / z + parameters.delta / (z - 1.0) + parameters.epsilon)
            * dh
            - ((parameters.alpha * z - parameters.q) / (z * (z - 1.0))) * h
        )
        return np.asarray((dh, d2h), dtype=np.complex128)

    solution = solve_ivp(
        rhs,
        (start, min(target_z)),
        np.asarray((h0, dh0), dtype=np.complex128),
        method="DOP853",
        t_eval=np.asarray(sorted(target_z, reverse=True)),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Heun integration failed: {solution.message}")
    by_z = {float(z): solution.y[:, i] for i, z in enumerate(solution.t)}
    return np.asarray([by_z[float(z)] for z in target_z])


def normalized_radial_modes(
    fixture: Fixture,
    spin_weight: int,
    radii: Sequence[float],
    **integration: float,
) -> np.ndarray:
    """Evaluate the specifically normalized Eq. (2.17a) horizon-in mode."""
    parameters = horizon_in_heun_parameters(fixture, spin_weight)
    sigma = fixture.sigma
    rp = fixture.r_plus
    rm = fixture.r_minus
    target_z = [-(r - rp) / sigma for r in radii]
    heun_values = integrate_heun(parameters, target_z, **integration)
    result = np.empty((len(radii), 2), dtype=np.complex128)
    for i, (r, (h, dh_dz)) in enumerate(zip(radii, heun_values)):
        x = r - rp
        prefactor = (
            sigma ** (-parameters.xi2 - spin_weight)
            * x ** (-parameters.xi1 - spin_weight)
            * (r - rm) ** parameters.xi2
            * np.exp(1j * fixture.omega * x)
        )
        logarithmic_derivative = (
            (-parameters.xi1 - spin_weight) / x
            + parameters.xi2 / (r - rm)
            + 1j * fixture.omega
        )
        result[i, 0] = prefactor * h
        result[i, 1] = prefactor * (logarithmic_derivative * h - dh_dz / sigma)
    return result


def jet_constant(value: Complex, degree: int) -> np.ndarray:
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
    if value[0] == 0.0:
        raise ZeroDivisionError("cannot invert a Taylor jet with zero value")
    result = np.zeros_like(value)
    result[0] = 1.0 / value[0]
    for n in range(1, len(value)):
        result[n] = -sum(value[k] * result[n - k] for k in range(1, n + 1)) / value[0]
    return result


def jet_divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return jet_multiply(left, jet_inverse(right))


def jet_derivative(value: np.ndarray) -> np.ndarray:
    return np.asarray([(n + 1) * value[n + 1] for n in range(len(value) - 1)])


def radial_solution_jet(
    fixture: Fixture,
    spin_weight: int,
    radius: float,
    value: Complex,
    derivative: Complex,
    degree: int,
) -> np.ndarray:
    """Generate on-shell Taylor derivatives from the independent radial ODE."""
    r = jet_variable(radius, degree)
    one = jet_constant(1.0, degree)
    delta = jet_multiply(r, r - 2.0 * fixture.mass * one)
    delta_prime = 2.0 * (r - fixture.mass * one)
    k = fixture.omega * jet_multiply(r, r)  # a=0 in this fixture
    inv_delta = jet_inverse(delta)
    a_coefficient = -(spin_weight + 1.0) * jet_multiply(delta_prime, inv_delta)
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
        rhs_coefficient = 0j
        for k_index in range(n + 1):
            rhs_coefficient += (
                a_coefficient[k_index] * (n - k_index + 1) * result[n - k_index + 1]
                + b_coefficient[k_index] * result[n - k_index]
            )
        result[n + 2] = rhs_coefficient / ((n + 2) * (n + 1))
    return result


def radial_operator(
    fixture: Fixture, radius: float, value: np.ndarray, *, dagger: bool
) -> np.ndarray:
    degree = len(value) - 1
    r = jet_variable(radius, degree)
    delta = jet_multiply(r, r - 2.0 * fixture.mass * jet_constant(1.0, degree))
    k_over_delta = jet_divide(fixture.omega * jet_multiply(r, r), delta)
    signed = 1j * k_over_delta if dagger else -1j * k_over_delta
    return jet_derivative(value) + jet_multiply(signed, value)[:-1]


def multiply_delta_squared(
    fixture: Fixture, radius: float, value: np.ndarray
) -> np.ndarray:
    degree = len(value) - 1
    r = jet_variable(radius, degree)
    delta = jet_multiply(r, r - 2.0 * fixture.mass * jet_constant(1.0, degree))
    return jet_multiply(jet_multiply(delta, delta), value)


def repeat_operator(
    fixture: Fixture,
    radius: float,
    value: np.ndarray,
    *,
    dagger: bool,
    count: int = 4,
) -> np.ndarray:
    for _ in range(count):
        value = radial_operator(fixture, radius, value, dagger=dagger)
    return value


@dataclass(frozen=True)
class Residuals:
    first_minus_to_plus: float
    first_plus_to_minus: float
    second_plus: float
    second_minus: float

    @property
    def maximum(self) -> float:
        return max(
            self.first_minus_to_plus,
            self.first_plus_to_minus,
            self.second_plus,
            self.second_minus,
        )


def relative_residual(actual: Complex, expected: Complex) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


def evaluate_identities(
    fixture: Fixture,
    radii: Sequence[float],
    plus_modes: np.ndarray,
    minus_modes: np.ndarray,
) -> list[Residuals]:
    sigma = fixture.sigma
    w = 4.0 * fixture.mass * fixture.omega * fixture.r_plus
    gamma = (w + 2j * sigma) * (w + 1j * sigma) * w * (w - 1j * sigma)
    angular_product = float(
        ((fixture.ell - 1) * fixture.ell * (fixture.ell + 1) * (fixture.ell + 2)) ** 2
    )
    c_product = angular_product + (12.0 * fixture.mass * fixture.omega) ** 2
    gamma_prime = c_product / gamma
    residuals: list[Residuals] = []
    for radius, plus, minus in zip(radii, plus_modes, minus_modes):
        plus_jet = radial_solution_jet(fixture, +2, radius, plus[0], plus[1], 8)
        minus_jet = radial_solution_jet(fixture, -2, radius, minus[0], minus[1], 8)

        minus_to_plus = repeat_operator(fixture, radius, minus_jet[:5], dagger=False)[0]
        plus_inner = multiply_delta_squared(fixture, radius, plus_jet[:5])
        plus_to_minus = multiply_delta_squared(
            fixture,
            radius,
            repeat_operator(fixture, radius, plus_inner, dagger=True),
        )[0]

        second_plus = repeat_operator(
            fixture,
            radius,
            multiply_delta_squared(
                fixture,
                radius,
                repeat_operator(
                    fixture,
                    radius,
                    multiply_delta_squared(fixture, radius, plus_jet),
                    dagger=True,
                ),
            ),
            dagger=False,
        )[0]
        second_minus = multiply_delta_squared(
            fixture,
            radius,
            repeat_operator(
                fixture,
                radius,
                multiply_delta_squared(
                    fixture,
                    radius,
                    repeat_operator(fixture, radius, minus_jet, dagger=False),
                ),
                dagger=True,
            ),
        )[0]
        residuals.append(
            Residuals(
                relative_residual(minus_to_plus, gamma * plus[0]),
                relative_residual(plus_to_minus, gamma_prime * minus[0]),
                relative_residual(second_plus, c_product * plus[0]),
                relative_residual(second_minus, c_product * minus[0]),
            )
        )
    return residuals


def maximum_mode_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.abs(right), 1.0e-300)
    return float(np.max(np.abs(left - right) / scale))


def run_fixture(verbose: bool = True) -> None:
    fixture = Fixture()
    # Stay a controlled distance from the singular Boyer--Lindquist
    # Kinnersley factors.  Horizon regularity belongs to the later tetrad and
    # hyperboloidal conversion gate, not this radial identity fixture.
    radii = (2.5, 3.0, 4.0, 7.0, 10.0)
    coarse_options = {
        "start": -2.0e-4,
        "series_order": 24,
        "rtol": 2.0e-10,
        "atol": 2.0e-12,
    }
    medium_options = {
        "start": -1.0e-4,
        "series_order": 32,
        "rtol": 2.0e-12,
        "atol": 2.0e-14,
    }
    fine_options = {
        "start": -5.0e-5,
        "series_order": 40,
        "rtol": 2.0e-13,
        "atol": 2.0e-15,
    }

    mode_sets: list[tuple[np.ndarray, np.ndarray]] = []
    residual_sets: list[list[Residuals]] = []
    for options in (coarse_options, medium_options, fine_options):
        plus = normalized_radial_modes(fixture, +2, radii, **options)
        minus = normalized_radial_modes(fixture, -2, radii, **options)
        mode_sets.append((plus, minus))
        residual_sets.append(evaluate_identities(fixture, radii, plus, minus))

    coarse_medium = max(
        maximum_mode_difference(mode_sets[0][0], mode_sets[1][0]),
        maximum_mode_difference(mode_sets[0][1], mode_sets[1][1]),
    )
    medium_fine = max(
        maximum_mode_difference(mode_sets[1][0], mode_sets[2][0]),
        maximum_mode_difference(mode_sets[1][1], mode_sets[2][1]),
    )
    maxima = [max(item.maximum for item in values) for values in residual_sets]

    if verbose:
        print("fixture M=1 a=0 omega=1/5 ell=m=2 horizon-in")
        for label, mode_difference, maximum in (
            ("coarse", float("nan"), maxima[0]),
            ("medium", coarse_medium, maxima[1]),
            ("fine", medium_fine, maxima[2]),
        ):
            print(
                f"{label:6s} max_RTSI_relative_residual={maximum:.6e} "
                f"previous_mode_difference={mode_difference:.6e}"
            )
        for radius, item in zip(radii, residual_sets[-1]):
            print(
                f"r={radius:4.1f} first(-to+)={item.first_minus_to_plus:.6e} "
                f"first(+to-)={item.first_plus_to_minus:.6e} "
                f"second(+)={item.second_plus:.6e} "
                f"second(-)={item.second_minus:.6e}"
            )

    if not medium_fine < coarse_medium:
        raise AssertionError(
            "normalized modes did not converge under tighter Frobenius/ODE controls: "
            f"coarse-medium={coarse_medium:.3e}, medium-fine={medium_fine:.3e}"
        )
    if maxima[-1] >= 2.0e-8:
        raise AssertionError(f"fine radial TSI residual {maxima[-1]:.3e} exceeds gate")
    print("PASS normalized Schwarzschild radial TSI first and second forms")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run_fixture(verbose=not args.quiet)


if __name__ == "__main__":
    main()
