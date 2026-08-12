#!/usr/bin/env python3
"""Independent rotating-Kerr coordinate oracle for linear Psi0.

The coordinate branch uses a local second-order multivariate jet type to
differentiate the tetrad-derived coordinate metric and a manufactured ORG
metric perturbation.  It varies Christoffels and the Riemann tensor directly;
no sigma/kappa expression occurs in that branch.  A separate NP branch is the
quantity under test and evaluates the corrected Campanelli--Lousto A5 form and
the inconsistent connection signs displayed in Loutrel et al. Eq. (12).

All geometry is the a != 0 code tetrad of Ripley et al. arXiv:2010.00162
Eqs. (5), (6), and (8), with signature +--- and Psi0=-C(l,m,l,m).
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
from dataclasses import dataclass


DIMENSION = 4
T_INDEX, R_INDEX, THETA_INDEX, PHI_INDEX = range(DIMENSION)
SQRT_TWO = math.sqrt(2.0)
I = 1.0j


def require_close(
    name: str,
    left: complex | float,
    right: complex | float,
    tolerance: float = 3.0e-11,
) -> None:
    error = abs(left - right)
    if error > tolerance:
        raise AssertionError(
            f"{name}: |left-right|={error:.17g} exceeds {tolerance:.3g}; "
            f"left={left}, right={right}"
        )
    print(f"PASS {name} error={error:.3e}")


def require_small(name: str, value: complex | float, tolerance: float) -> None:
    require_close(name, value, 0.0, tolerance)


def require_separated(
    name: str,
    left: complex,
    right: complex,
    minimum: float = 2.0e-6,
) -> None:
    separation = abs(left - right)
    if separation < minimum:
        raise AssertionError(
            f"{name}: separation={separation:.17g} is below {minimum:.3g}"
        )
    print(f"PASS {name} separation={separation:.3e}")


def require_nonzero(name: str, values: list[complex], minimum: float) -> None:
    magnitude = max(abs(value) for value in values)
    if magnitude < minimum:
        raise AssertionError(
            f"{name}: maximum magnitude={magnitude:.17g} is below {minimum:.3g}"
        )
    print(f"PASS {name} magnitude={magnitude:.3e}")


@dataclass(frozen=True)
class Jet2:
    """Value, gradient, and Hessian with respect to four real coordinates."""

    value: complex
    gradient: tuple[complex, ...]
    hessian: tuple[tuple[complex, ...], ...]

    @staticmethod
    def constant(value: complex | float) -> Jet2:
        return Jet2(
            complex(value),
            (0.0j,) * DIMENSION,
            tuple((0.0j,) * DIMENSION for _ in range(DIMENSION)),
        )

    @staticmethod
    def variable(value: float, index: int) -> Jet2:
        gradient = [0.0j] * DIMENSION
        gradient[index] = 1.0 + 0.0j
        return Jet2.constant(value)._replace_gradient(tuple(gradient))

    def _replace_gradient(self, gradient: tuple[complex, ...]) -> Jet2:
        return Jet2(self.value, gradient, self.hessian)

    @staticmethod
    def coerce(value: Jet2 | complex | float) -> Jet2:
        return value if isinstance(value, Jet2) else Jet2.constant(value)

    def unary(self, value: complex, first: complex, second: complex) -> Jet2:
        gradient = tuple(first * self.gradient[i] for i in range(DIMENSION))
        hessian = tuple(
            tuple(
                second * self.gradient[i] * self.gradient[j]
                + first * self.hessian[i][j]
                for j in range(DIMENSION)
            )
            for i in range(DIMENSION)
        )
        return Jet2(value, gradient, hessian)

    def conjugate(self) -> Jet2:
        return Jet2(
            self.value.conjugate(),
            tuple(value.conjugate() for value in self.gradient),
            tuple(
                tuple(value.conjugate() for value in row)
                for row in self.hessian
            ),
        )

    def __add__(self, other: Jet2 | complex | float) -> Jet2:
        right = Jet2.coerce(other)
        return Jet2(
            self.value + right.value,
            tuple(self.gradient[i] + right.gradient[i]
                  for i in range(DIMENSION)),
            tuple(
                tuple(self.hessian[i][j] + right.hessian[i][j]
                      for j in range(DIMENSION))
                for i in range(DIMENSION)
            ),
        )

    __radd__ = __add__

    def __neg__(self) -> Jet2:
        return self * -1.0

    def __sub__(self, other: Jet2 | complex | float) -> Jet2:
        return self + (-Jet2.coerce(other))

    def __rsub__(self, other: Jet2 | complex | float) -> Jet2:
        return Jet2.coerce(other) - self

    def __mul__(self, other: Jet2 | complex | float) -> Jet2:
        right = Jet2.coerce(other)
        gradient = tuple(
            self.gradient[i] * right.value
            + self.value * right.gradient[i]
            for i in range(DIMENSION)
        )
        hessian = tuple(
            tuple(
                self.hessian[i][j] * right.value
                + self.gradient[i] * right.gradient[j]
                + self.gradient[j] * right.gradient[i]
                + self.value * right.hessian[i][j]
                for j in range(DIMENSION)
            )
            for i in range(DIMENSION)
        )
        return Jet2(self.value * right.value, gradient, hessian)

    __rmul__ = __mul__

    def reciprocal(self) -> Jet2:
        return self.unary(
            1.0 / self.value,
            -1.0 / self.value**2,
            2.0 / self.value**3,
        )

    def __truediv__(self, other: Jet2 | complex | float) -> Jet2:
        return self * Jet2.coerce(other).reciprocal()

    def __rtruediv__(self, other: Jet2 | complex | float) -> Jet2:
        return Jet2.coerce(other) / self

    def __pow__(self, exponent: int) -> Jet2:
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return (self.reciprocal()) ** (-exponent)
        result = Jet2.constant(1.0)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result


def jet_sin(value: Jet2) -> Jet2:
    return value.unary(cmath.sin(value.value), cmath.cos(value.value),
                       -cmath.sin(value.value))


def jet_cos(value: Jet2) -> Jet2:
    return value.unary(cmath.cos(value.value), -cmath.sin(value.value),
                       -cmath.cos(value.value))


def jet_exp(value: Jet2) -> Jet2:
    exponential = cmath.exp(value.value)
    return value.unary(exponential, exponential, exponential)


@dataclass(frozen=True)
class FirstJet:
    value: complex
    gradient: tuple[complex, ...]

    @staticmethod
    def coerce(value: FirstJet | complex | float) -> FirstJet:
        if isinstance(value, FirstJet):
            return value
        return FirstJet(complex(value), (0.0j,) * DIMENSION)

    def __add__(self, other: FirstJet | complex | float) -> FirstJet:
        right = FirstJet.coerce(other)
        return FirstJet(
            self.value + right.value,
            tuple(self.gradient[i] + right.gradient[i]
                  for i in range(DIMENSION)),
        )

    __radd__ = __add__

    def __neg__(self) -> FirstJet:
        return self * -1.0

    def __sub__(self, other: FirstJet | complex | float) -> FirstJet:
        return self + (-FirstJet.coerce(other))

    def __rsub__(self, other: FirstJet | complex | float) -> FirstJet:
        return FirstJet.coerce(other) - self

    def __mul__(self, other: FirstJet | complex | float) -> FirstJet:
        right = FirstJet.coerce(other)
        return FirstJet(
            self.value * right.value,
            tuple(
                self.gradient[i] * right.value
                + self.value * right.gradient[i]
                for i in range(DIMENSION)
            ),
        )

    __rmul__ = __mul__


def first_jet(value: Jet2) -> FirstJet:
    return FirstJet(value.value, value.gradient)


def directional_first(vector: list[Jet2], scalar: Jet2) -> FirstJet:
    return FirstJet(
        sum(vector[i].value * scalar.gradient[i] for i in range(DIMENSION)),
        tuple(
            sum(
                vector[i].gradient[derivative] * scalar.gradient[i]
                + vector[i].value * scalar.hessian[derivative][i]
                for i in range(DIMENSION)
            )
            for derivative in range(DIMENSION)
        ),
    )


def directional_value(vector: list[Jet2], scalar: FirstJet) -> complex:
    return sum(
        vector[i].value * scalar.gradient[i] for i in range(DIMENSION)
    )


JetMatrix = list[list[Jet2]]
ComplexMatrix = list[list[complex]]


def zero_jet_matrix() -> JetMatrix:
    return [
        [Jet2.constant(0.0) for _ in range(DIMENSION)]
        for _ in range(DIMENSION)
    ]


def outer(left: list[Jet2], right: list[Jet2]) -> JetMatrix:
    return [
        [left[row] * right[column] for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def matrix_add(*matrices: JetMatrix) -> JetMatrix:
    return [
        [sum((matrix[row][column] for matrix in matrices), Jet2.constant(0.0))
         for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def matrix_scale(factor: complex | float | Jet2, matrix: JetMatrix) -> JetMatrix:
    return [
        [factor * matrix[row][column] for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def matrix_vector(matrix: JetMatrix, vector: list[Jet2]) -> list[Jet2]:
    return [
        sum((matrix[row][column] * vector[column]
             for column in range(DIMENSION)), Jet2.constant(0.0))
        for row in range(DIMENSION)
    ]


def invert_jet_matrix(matrix: JetMatrix) -> JetMatrix:
    augmented = [
        list(matrix[row])
        + [Jet2.constant(1.0 if row == column else 0.0)
           for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    for column in range(DIMENSION):
        pivot = max(range(column, DIMENSION),
                    key=lambda row: abs(augmented[row][column].value))
        if abs(augmented[pivot][column].value) < 1.0e-14:
            raise AssertionError("singular jet matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse_pivot = augmented[column][column].reciprocal()
        augmented[column] = [entry * inverse_pivot for entry in augmented[column]]
        for row in range(DIMENSION):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][entry] - factor * augmented[column][entry]
                for entry in range(2 * DIMENSION)
            ]
    return [row[DIMENSION:] for row in augmented]


def invert_complex_matrix(matrix: ComplexMatrix) -> ComplexMatrix:
    augmented = [
        list(matrix[row])
        + [complex(1.0 if row == column else 0.0)
           for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    for column in range(DIMENSION):
        pivot = max(range(column, DIMENSION),
                    key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise AssertionError("singular complex matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [entry / pivot_value for entry in augmented[column]]
        for row in range(DIMENSION):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][entry] - factor * augmented[column][entry]
                for entry in range(2 * DIMENSION)
            ]
    return [row[DIMENSION:] for row in augmented]


def contract(left: list[Jet2], matrix: JetMatrix, right: list[Jet2]) -> Jet2:
    return sum(
        (left[row] * matrix[row][column] * right[column]
         for row in range(DIMENSION)
         for column in range(DIMENSION)),
        Jet2.constant(0.0),
    )


@dataclass(frozen=True)
class KerrGeometry:
    l: list[Jet2]
    n: list[Jet2]
    m: list[Jet2]
    mbar: list[Jet2]
    inverse_metric: JetMatrix
    metric: JetMatrix
    rho: Jet2
    epsilon: Jet2
    alpha: Jet2
    beta: Jet2
    tau: Jet2
    pi: Jet2


def kerr_geometry(
    coordinates: tuple[Jet2, Jet2, Jet2, Jet2],
    mass: float,
    spin: float,
    length: float,
) -> KerrGeometry:
    _, radius, theta, _ = coordinates
    sin_theta = jet_sin(theta)
    cos_theta = jet_cos(theta)
    length2 = length**2
    length4 = length**4
    spin2 = spin**2
    denominator = length4 + spin2 * radius**2 * cos_theta**2
    radial_polynomial = (
        length2 - 2.0 * mass * radius + (spin / length) ** 2 * radius**2
    )
    l = [
        radius**2 / denominator
        * (2.0 * mass * (2.0 * mass - (spin / length) ** 2 * radius)),
        -0.5 * radius**2 / denominator * radial_polynomial,
        Jet2.constant(0.0),
        spin * radius**2 / denominator,
    ]
    n = [
        2.0 + 4.0 * mass * radius / length2,
        radius**2 / length2,
        Jet2.constant(0.0),
        Jet2.constant(0.0),
    ]
    angular_denominator = length2 - I * spin * radius * cos_theta
    angular_prefactor = radius / (SQRT_TWO * angular_denominator)
    m = [
        -I * spin * sin_theta * angular_prefactor,
        Jet2.constant(0.0),
        -angular_prefactor,
        -I * angular_prefactor / sin_theta,
    ]
    mbar = [component.conjugate() for component in m]
    inverse_metric = matrix_add(
        outer(l, n),
        outer(n, l),
        matrix_scale(-1.0, outer(m, mbar)),
        matrix_scale(-1.0, outer(mbar, m)),
    )
    metric = invert_jet_matrix(inverse_metric)

    dm = length2 - I * spin * radius * cos_theta
    dp = length2 + I * spin * radius * cos_theta
    real_denominator = length4 + spin2 * radius**2 * cos_theta**2
    rho = -radius * (
        spin2 * radius**2 + length4 - 2.0 * length2 * mass * radius
    ) / (2.0 * dm**2 * dp)
    epsilon = radius**2 * (
        length2 * mass
        - spin2 * radius
        - I * spin * (length2 - mass * radius) * cos_theta
    ) / (2.0 * dm**2 * dp)
    alpha = radius * cos_theta / (2.0 * SQRT_TWO * sin_theta * dp)
    beta = radius * (
        -length2 * cos_theta / sin_theta
        + I * spin * radius * sin_theta * (1.0 / sin_theta**2 + 1.0)
    ) / (2.0 * SQRT_TWO * dm**2)
    tau = I * spin * radius**2 * sin_theta / (SQRT_TWO * dm**2)
    pi = -I * spin * radius**2 * sin_theta / (
        SQRT_TWO * real_denominator
    )
    return KerrGeometry(l, n, m, mbar, inverse_metric, metric, rho, epsilon,
                        alpha, beta, tau, pi)


def matrix_value(matrix: JetMatrix) -> ComplexMatrix:
    return [[entry.value for entry in row] for row in matrix]


def commutator(left: list[Jet2], right: list[Jet2]) -> list[complex]:
    return [
        sum(
            left[index].value * right[component].gradient[index]
            - right[index].value * left[component].gradient[index]
            for index in range(DIMENSION)
        )
        for component in range(DIMENSION)
    ]


def matrix_vector_complex(
    matrix: ComplexMatrix, vector: list[complex]
) -> list[complex]:
    return [
        sum(matrix[row][column] * vector[column]
            for column in range(DIMENSION))
        for row in range(DIMENSION)
    ]


def validate_geometry(name: str, geometry: KerrGeometry) -> None:
    require_close(f"{name} tetrad l dot n", contract(
        geometry.l, geometry.metric, geometry.n).value, 1.0)
    require_close(f"{name} tetrad m dot mbar", contract(
        geometry.m, geometry.metric, geometry.mbar).value, -1.0)
    remaining = [
        contract(left, geometry.metric, right).value
        for left, right in (
            (geometry.l, geometry.l),
            (geometry.n, geometry.n),
            (geometry.m, geometry.m),
            (geometry.l, geometry.m),
            (geometry.l, geometry.mbar),
            (geometry.n, geometry.m),
            (geometry.n, geometry.mbar),
        )
    ]
    require_small(f"{name} remaining tetrad products", max(map(abs, remaining)),
                  2.0e-12)

    product_residual = []
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            product = sum(
                geometry.inverse_metric[row][inner].value
                * geometry.metric[inner][column].value
                for inner in range(DIMENSION)
            )
            product_residual.append(
                product - (1.0 if row == column else 0.0)
            )
    require_small(f"{name} metric inverse", max(map(abs, product_residual)),
                  2.0e-12)

    tetrad_columns = [geometry.l, geometry.n, geometry.m, geometry.mbar]
    tetrad_matrix = [
        [tetrad_columns[column][row].value for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    inverse_tetrad = invert_complex_matrix(tetrad_matrix)
    delta_d = matrix_vector_complex(
        inverse_tetrad, commutator(geometry.n, geometry.l)
    )
    delta_small_d = matrix_vector_complex(
        inverse_tetrad, commutator(geometry.m, geometry.l)
    )
    delta_bar_delta = matrix_vector_complex(
        inverse_tetrad, commutator(geometry.mbar, geometry.m)
    )
    rho = geometry.rho.value
    epsilon = geometry.epsilon.value
    alpha = geometry.alpha.value
    beta = geometry.beta.value
    tau = geometry.tau.value
    pi = geometry.pi.value
    expected_delta_d = [
        0.0j,
        epsilon + epsilon.conjugate(),
        -(tau.conjugate() + pi),
        -(tau + pi.conjugate()),
    ]
    expected_delta_small_d = [
        alpha.conjugate() + beta - pi.conjugate(),
        0.0j,
        -(rho.conjugate() + epsilon - epsilon.conjugate()),
        0.0j,
    ]
    # Only the angular coefficients of [delta_bar,delta] are needed here;
    # the radial ones involve mu, which T0 does not consume.
    expected_angular = [
        alpha - beta.conjugate(),
        beta - alpha.conjugate(),
    ]
    residual = [
        delta_d[index] - expected_delta_d[index]
        for index in range(DIMENSION)
    ] + [
        delta_small_d[index] - expected_delta_small_d[index]
        for index in range(DIMENSION)
    ] + [
        delta_bar_delta[2 + index] - expected_angular[index]
        for index in range(2)
    ]
    require_small(f"{name} Eq8 coefficients reproduce tetrad commutators",
                  max(map(abs, residual)), 2.0e-11)


def reconstruct_org_metric(
    geometry: KerrGeometry, h_ll: Jet2, h_lm: Jet2, h_mm: Jet2
) -> JetMatrix:
    n_covector = matrix_vector(geometry.metric, geometry.n)
    m_covector = matrix_vector(geometry.metric, geometry.m)
    mbar_covector = matrix_vector(geometry.metric, geometry.mbar)
    return matrix_add(
        matrix_scale(h_ll, outer(n_covector, n_covector)),
        matrix_scale(-h_lm, matrix_add(
            outer(n_covector, mbar_covector),
            outer(mbar_covector, n_covector),
        )),
        matrix_scale(-h_lm.conjugate(), matrix_add(
            outer(n_covector, m_covector),
            outer(m_covector, n_covector),
        )),
        matrix_scale(h_mm, outer(mbar_covector, mbar_covector)),
        matrix_scale(h_mm.conjugate(), outer(m_covector, m_covector)),
    )


def validate_org_metric(
    name: str,
    geometry: KerrGeometry,
    perturbation: JetMatrix,
    h_ll: Jet2,
    h_lm: Jet2,
    h_mm: Jet2,
) -> None:
    require_close(f"{name} ORG h_ll projection",
                  contract(geometry.l, perturbation, geometry.l).value,
                  h_ll.value)
    require_close(f"{name} ORG h_lm projection",
                  contract(geometry.l, perturbation, geometry.m).value,
                  h_lm.value)
    require_close(f"{name} ORG h_mm projection",
                  contract(geometry.m, perturbation, geometry.m).value,
                  h_mm.value)
    n_contraction = matrix_vector(perturbation, geometry.n)
    require_small(f"{name} ORG n contraction",
                  max(abs(value.value) for value in n_contraction), 3.0e-11)
    trace = sum(
        geometry.inverse_metric[row][column].value
        * perturbation[row][column].value
        for row in range(DIMENSION)
        for column in range(DIMENSION)
    )
    require_small(f"{name} ORG trace", trace, 3.0e-11)
    reality_residuals: list[float] = []
    symmetry_residuals: list[float] = []
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            entry = perturbation[row][column]
            transpose = perturbation[column][row]
            reality_residuals.extend(
                [abs(entry.value.imag)]
                + [abs(value.imag) for value in entry.gradient]
                + [abs(value.imag) for hessian_row in entry.hessian
                   for value in hessian_row]
            )
            symmetry_residuals.extend(
                [abs(entry.value - transpose.value)]
                + [abs(entry.gradient[index] - transpose.gradient[index])
                   for index in range(DIMENSION)]
                + [abs(entry.hessian[first][second]
                       - transpose.hessian[first][second])
                   for first in range(DIMENSION)
                   for second in range(DIMENSION)]
            )
    require_small(f"{name} real symmetric coordinate perturbation jets",
                  max(reality_residuals + symmetry_residuals), 3.0e-11)


def coordinate_linearized_psi0(
    geometry: KerrGeometry, perturbation: JetMatrix
) -> tuple[complex, float, complex]:
    g = matrix_value(geometry.metric)
    h = matrix_value(perturbation)
    inverse_g = matrix_value(geometry.inverse_metric)
    dg = [
        [[geometry.metric[row][column].gradient[derivative]
          for column in range(DIMENSION)] for row in range(DIMENSION)]
        for derivative in range(DIMENSION)
    ]
    dh = [
        [[perturbation[row][column].gradient[derivative]
          for column in range(DIMENSION)] for row in range(DIMENSION)]
        for derivative in range(DIMENSION)
    ]
    ddg = [
        [[[geometry.metric[row][column].hessian[first][second]
           for column in range(DIMENSION)] for row in range(DIMENSION)]
         for second in range(DIMENSION)]
        for first in range(DIMENSION)
    ]
    ddh = [
        [[[perturbation[row][column].hessian[first][second]
           for column in range(DIMENSION)] for row in range(DIMENSION)]
         for second in range(DIMENSION)]
        for first in range(DIMENSION)
    ]
    derivative_inverse_g = [
        [[geometry.inverse_metric[row][column].gradient[derivative]
          for column in range(DIMENSION)] for row in range(DIMENSION)]
        for derivative in range(DIMENSION)
    ]

    metric_sum = [
        [[dg[b][d][c] + dg[c][d][b] - dg[d][b][c]
          for c in range(DIMENSION)] for b in range(DIMENSION)]
        for d in range(DIMENSION)
    ]
    derivative_metric_sum = [
        [[[ddg[e][b][d][c] + ddg[e][c][d][b] - ddg[e][d][b][c]
           for c in range(DIMENSION)] for b in range(DIMENSION)]
         for d in range(DIMENSION)] for e in range(DIMENSION)
    ]
    perturbation_sum = [
        [[dh[b][d][c] + dh[c][d][b] - dh[d][b][c]
          for c in range(DIMENSION)] for b in range(DIMENSION)]
        for d in range(DIMENSION)
    ]
    derivative_perturbation_sum = [
        [[[ddh[e][b][d][c] + ddh[e][c][d][b] - ddh[e][d][b][c]
           for c in range(DIMENSION)] for b in range(DIMENSION)]
         for d in range(DIMENSION)] for e in range(DIMENSION)
    ]

    christoffel = [
        [[0.5 * sum(inverse_g[a][d] * metric_sum[d][b][c]
                    for d in range(DIMENSION))
          for c in range(DIMENSION)] for b in range(DIMENSION)]
        for a in range(DIMENSION)
    ]
    derivative_christoffel = [
        [[[0.5 * sum(
            derivative_inverse_g[e][a][d] * metric_sum[d][b][c]
            + inverse_g[a][d] * derivative_metric_sum[e][d][b][c]
            for d in range(DIMENSION))
           for c in range(DIMENSION)] for b in range(DIMENSION)]
         for a in range(DIMENSION)] for e in range(DIMENSION)
    ]
    delta_inverse_g = [
        [-sum(inverse_g[a][p] * h[p][q] * inverse_g[q][d]
              for p in range(DIMENSION) for q in range(DIMENSION))
         for d in range(DIMENSION)] for a in range(DIMENSION)
    ]
    derivative_delta_inverse_g = [
        [[-sum(
            derivative_inverse_g[e][a][p] * h[p][q] * inverse_g[q][d]
            + inverse_g[a][p] * dh[e][p][q] * inverse_g[q][d]
            + inverse_g[a][p] * h[p][q] * derivative_inverse_g[e][q][d]
            for p in range(DIMENSION) for q in range(DIMENSION))
          for d in range(DIMENSION)] for a in range(DIMENSION)]
        for e in range(DIMENSION)
    ]
    delta_christoffel = [
        [[0.5 * sum(
            delta_inverse_g[a][d] * metric_sum[d][b][c]
            + inverse_g[a][d] * perturbation_sum[d][b][c]
            for d in range(DIMENSION))
          for c in range(DIMENSION)] for b in range(DIMENSION)]
        for a in range(DIMENSION)
    ]
    derivative_delta_christoffel = [
        [[[0.5 * sum(
            derivative_delta_inverse_g[e][a][d] * metric_sum[d][b][c]
            + delta_inverse_g[a][d] * derivative_metric_sum[e][d][b][c]
            + derivative_inverse_g[e][a][d] * perturbation_sum[d][b][c]
            + inverse_g[a][d] * derivative_perturbation_sum[e][d][b][c]
            for d in range(DIMENSION))
           for c in range(DIMENSION)] for b in range(DIMENSION)]
         for a in range(DIMENSION)] for e in range(DIMENSION)
    ]

    def riemann_up(a: int, b: int, c: int, d: int) -> complex:
        return (
            derivative_christoffel[c][a][d][b]
            - derivative_christoffel[d][a][c][b]
            + sum(
                christoffel[a][c][e] * christoffel[e][d][b]
                - christoffel[a][d][e] * christoffel[e][c][b]
                for e in range(DIMENSION)
            )
        )

    def delta_riemann_up(a: int, b: int, c: int, d: int) -> complex:
        return (
            derivative_delta_christoffel[c][a][d][b]
            - derivative_delta_christoffel[d][a][c][b]
            + sum(
                delta_christoffel[a][c][e] * christoffel[e][d][b]
                + christoffel[a][c][e] * delta_christoffel[e][d][b]
                - delta_christoffel[a][d][e] * christoffel[e][c][b]
                - christoffel[a][d][e] * delta_christoffel[e][c][b]
                for e in range(DIMENSION)
            )
        )

    maximum_ricci = 0.0
    for b in range(DIMENSION):
        for d in range(DIMENSION):
            ricci = sum(riemann_up(a, b, a, d) for a in range(DIMENSION))
            maximum_ricci = max(maximum_ricci, abs(ricci))

    l = [entry.value for entry in geometry.l]
    m = [entry.value for entry in geometry.m]
    background_contraction = 0.0j
    perturbation_contraction = 0.0j
    for a, b, c, d in itertools.product(range(DIMENSION), repeat=4):
        riemann_down = sum(
            g[a][e] * riemann_up(e, b, c, d) for e in range(DIMENSION)
        )
        delta_riemann_down = sum(
            h[a][e] * riemann_up(e, b, c, d)
            + g[a][e] * delta_riemann_up(e, b, c, d)
            for e in range(DIMENSION)
        )
        tetrad_product = l[a] * m[b] * l[c] * m[d]
        background_contraction += tetrad_product * riemann_down
        perturbation_contraction += tetrad_product * delta_riemann_down

    return -perturbation_contraction, maximum_ricci, -background_contraction


def np_linear_psi0(
    geometry: KerrGeometry, h_ll: Jet2, h_lm: Jet2, h_mm: Jet2
) -> tuple[complex, complex]:
    rho = first_jet(geometry.rho)
    epsilon = first_jet(geometry.epsilon)
    alpha = first_jet(geometry.alpha)
    beta = first_jet(geometry.beta)
    tau = first_jet(geometry.tau)
    pi = first_jet(geometry.pi)
    rho_bar = first_jet(geometry.rho.conjugate())
    epsilon_bar = first_jet(geometry.epsilon.conjugate())
    alpha_bar = first_jet(geometry.alpha.conjugate())
    pi_bar = first_jet(geometry.pi.conjugate())

    sigma = (
        0.5
        * (
            directional_first(geometry.l, h_mm)
            + (2.0 * (epsilon_bar - epsilon) + rho - rho_bar)
            * first_jet(h_mm)
        )
        - (tau + pi_bar) * first_jet(h_lm)
    )
    kappa = (
        directional_first(geometry.l, h_lm)
        - (2.0 * epsilon + rho_bar) * first_jet(h_lm)
        - 0.5
        * (
            directional_first(geometry.m, h_ll)
            + (-2.0 * alpha_bar - 2.0 * beta + pi_bar + tau)
            * first_jet(h_ll)
        )
    )
    radial = (
        directional_value(geometry.l, sigma)
        + (-rho - rho_bar - 3.0 * epsilon + epsilon_bar).value
        * sigma.value
    )
    corrected_angular = (
        directional_value(geometry.m, kappa)
        + (-alpha_bar - 3.0 * beta + pi_bar - tau).value * kappa.value
    )
    old_displayed_angular = (
        directional_value(geometry.m, kappa)
        + (alpha_bar + 3.0 * beta - pi_bar + tau).value * kappa.value
    )
    return radial - corrected_angular, radial - old_displayed_angular


def random_coefficient(rng: random.Random) -> float:
    value = rng.uniform(-0.8, 0.8)
    return value if abs(value) > 0.08 else value + math.copysign(0.13, value or 1.0)


def random_amplitude(
    coordinates: tuple[Jet2, Jet2, Jet2, Jet2],
    rng: random.Random,
    complex_field: bool,
) -> Jet2:
    time, radius, theta, phi = coordinates
    sin_theta = jet_sin(theta)
    cos_theta = jet_cos(theta)
    basis = [
        Jet2.constant(1.0),
        time,
        radius,
        sin_theta,
        time**2,
        radius**2,
        time * radius,
        time * sin_theta,
        radius * cos_theta,
        sin_theta * jet_cos(phi),
        cos_theta * jet_sin(phi),
    ]
    real_part = sum(
        (random_coefficient(rng) * term for term in basis),
        Jet2.constant(0.0),
    )
    if not complex_field:
        return real_part
    imaginary_part = sum(
        (random_coefficient(rng) * term for term in reversed(basis)),
        Jet2.constant(0.0),
    )
    return real_part + I * imaginary_part


def manufactured_fields(
    coordinates: tuple[Jet2, Jet2, Jet2, Jet2],
    seed: int,
    mode: int,
) -> tuple[Jet2, Jet2, Jet2]:
    rng = random.Random(seed)
    phase = jet_exp(I * mode * coordinates[PHI_INDEX])
    h_ll = random_amplitude(coordinates, rng, False)
    h_lm = random_amplitude(coordinates, rng, True) * phase
    h_mm = random_amplitude(coordinates, rng, True) * phase**2
    return h_ll, h_lm, h_mm


def validate_derivative_coverage(
    name: str, fields: tuple[Jet2, Jet2, Jet2]
) -> None:
    coordinate_names = ("time", "radial", "polar", "azimuthal")
    for derivative, coordinate_name in enumerate(coordinate_names):
        require_nonzero(
            f"{name} {coordinate_name} first-derivative coverage",
            [field.gradient[derivative] for field in fields],
            2.0e-3,
        )
        require_nonzero(
            f"{name} {coordinate_name} second-derivative coverage",
            [field.hessian[derivative][derivative] for field in fields],
            2.0e-3,
        )
    require_nonzero(
        f"{name} mixed time-radial derivative coverage",
        [field.hessian[T_INDEX][R_INDEX] for field in fields],
        2.0e-3,
    )
    require_nonzero(
        f"{name} mixed polar-azimuthal derivative coverage",
        [field.hessian[THETA_INDEX][PHI_INDEX] for field in fields],
        2.0e-3,
    )


@dataclass(frozen=True)
class Fixture:
    name: str
    mass: float
    spin: float
    length: float
    point: tuple[float, float, float, float]
    seed: int
    mode: int


def fixtures() -> list[Fixture]:
    return [
        Fixture("moderate_spin", 1.0, 0.63, 2.0,
                (0.17, 0.72, 0.91, 0.38), 2026081101, 1),
        Fixture("rapid_spin", 1.0, 0.91, 2.0,
                (-0.23, 1.08, 1.17, -0.44), 2026081102, 2),
        Fixture("near_extremal", 1.0, 0.999, 2.0,
                (-0.11, 1.35, 1.39, 0.21), 2026081104, -2),
        Fixture("opposite_spin", 1.0, -0.74, 2.0,
                (0.31, 0.57, 0.73, 0.62), 2026081103, 3),
    ]


def validate_jet_algebra() -> None:
    point = (0.21, 0.68, 0.87, -0.33)
    variables = tuple(Jet2.variable(point[i], i) for i in range(DIMENSION))
    time, radius, theta, phi = variables
    expression = jet_exp(I * phi) * jet_sin(theta) / (1.3 + radius * time)
    denominator = 1.3 + point[R_INDEX] * point[T_INDEX]
    angular_factor = cmath.exp(I * point[PHI_INDEX])
    expected_value = (
        angular_factor * math.sin(point[THETA_INDEX]) / denominator
    )
    expected_mixed_tr = (
        -angular_factor
        * math.sin(point[THETA_INDEX])
        * (1.3 - point[R_INDEX] * point[T_INDEX])
        / denominator**3
    )
    require_close("jet analytic azimuthal first derivative",
                  expression.gradient[PHI_INDEX], I * expected_value,
                  2.0e-14)
    require_close("jet analytic polar second derivative",
                  expression.hessian[THETA_INDEX][THETA_INDEX],
                  -expected_value, 2.0e-14)
    require_close("jet analytic azimuthal second derivative",
                  expression.hessian[PHI_INDEX][PHI_INDEX],
                  -expected_value, 2.0e-14)
    require_close("jet analytic mixed time-radial derivative",
                  expression.hessian[T_INDEX][R_INDEX], expected_mixed_tr,
                  2.0e-13)
    require_close("jet Hessian symmetry",
                  expression.hessian[T_INDEX][THETA_INDEX],
                  expression.hessian[THETA_INDEX][T_INDEX], 2.0e-14)


def main() -> None:
    validate_jet_algebra()
    for fixture in fixtures():
        boyer_lindquist_radius = fixture.length**2 / fixture.point[R_INDEX]
        outer_horizon = fixture.mass + math.sqrt(
            fixture.mass**2 - fixture.spin**2
        )
        require_nonzero(f"{fixture.name} nonzero Kerr spin", [fixture.spin],
                        1.0e-2)
        if boyer_lindquist_radius <= outer_horizon:
            raise AssertionError(
                f"{fixture.name}: point r={boyer_lindquist_radius} is not "
                f"outside r+={outer_horizon}"
            )
        print(
            f"PASS {fixture.name} exterior point "
            f"r/r_plus={boyer_lindquist_radius / outer_horizon:.3f}"
        )
        coordinates = tuple(
            Jet2.variable(fixture.point[index], index)
            for index in range(DIMENSION)
        )
        geometry = kerr_geometry(
            coordinates, fixture.mass, fixture.spin, fixture.length
        )
        validate_geometry(fixture.name, geometry)
        fields = manufactured_fields(coordinates, fixture.seed, fixture.mode)
        validate_derivative_coverage(fixture.name, fields)
        perturbation = reconstruct_org_metric(geometry, *fields)
        validate_org_metric(fixture.name, geometry, perturbation, *fields)
        coordinate_value, maximum_ricci, background_psi0 = (
            coordinate_linearized_psi0(geometry, perturbation)
        )
        corrected_value, old_value = np_linear_psi0(geometry, *fields)
        require_small(f"{fixture.name} background Ricci tensor",
                      maximum_ricci, 3.0e-11)
        require_small(f"{fixture.name} background Psi0", background_psi0,
                      2.0e-11)
        require_close(
            f"{fixture.name} coordinate Weyl equals corrected NP T0",
            coordinate_value,
            corrected_value,
            3.0e-11,
        )
        require_separated(
            f"{fixture.name} coordinate Weyl rejects displayed Eq12 signs",
            coordinate_value,
            old_value,
        )

    print("Completed rotating-Kerr numerical AD linear Psi0 checks")


if __name__ == "__main__":
    main()
