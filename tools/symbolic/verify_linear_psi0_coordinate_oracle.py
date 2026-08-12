#!/usr/bin/env python3
"""Independent coordinate-tensor oracle for linear Psi0 in repository conventions.

The coordinate route varies the metric Christoffels and Riemann tensor
directly.  It does not use the NP sigma/kappa algebra.  Only the final
fixed-background-tetrad contraction is compared with the corrected NP path.

To keep the audit transparent and fast, this uses the Schwarzschild member
(a=0, M=1, L=2) of Ripley et al. arXiv:2010.00162 Eqs. (5)--(6).  This is
still a curved background in the repository's horizon-penetrating,
hyperboloidal compact coordinates, with nonzero rho, epsilon, alpha, and beta.
It therefore distinguishes the corrected angular connection signs from the
incorrect displayed signs in Loutrel et al. arXiv:2008.11770 Eq. (12).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import sympy as sp


DIMENSION = 4
T, R, THETA, PHI = sp.symbols("T R theta phi", real=True)
COORDINATES = (T, R, THETA, PHI)
SQRT_TWO = sp.sqrt(2)
I = sp.I


def require_symbolic_zero(name: str, expression: sp.Expr) -> None:
    reduced = sp.trigsimp(
        sp.factor(sp.cancel(sp.expand_complex(expression).rewrite(sp.exp)))
    )
    if reduced != 0:
        raise AssertionError(f"{name}: expected zero, got {reduced}")
    print(f"PASS {name}")


def require_symbolic_vector_zero(name: str, vector: sp.Matrix) -> None:
    for component in vector:
        reduced = sp.trigsimp(
            sp.factor(sp.cancel(sp.expand_complex(component).rewrite(sp.exp)))
        )
        if reduced != 0:
            raise AssertionError(f"{name}: expected zero, got {reduced}")
    print(f"PASS {name}")


def require_close(name: str, left: complex, right: complex,
                  tolerance: float = 2.0e-12) -> None:
    error = abs(left - right)
    if error > tolerance:
        raise AssertionError(
            f"{name}: |left-right|={error:.17g} exceeds {tolerance:.3g}; "
            f"left={left}, right={right}"
        )
    print(f"PASS {name} error={error:.3e}")


def require_separated(name: str, left: complex, right: complex,
                      minimum: float = 1.0e-6) -> None:
    separation = abs(left - right)
    if separation < minimum:
        raise AssertionError(
            f"{name}: separation={separation:.17g} is below {minimum:.3g}"
        )
    print(f"PASS {name} separation={separation:.3e}")


def directional(vector: sp.Matrix, expression: sp.Expr) -> sp.Expr:
    return sp.expand(
        sum(
            vector[index] * sp.diff(expression, COORDINATES[index])
            for index in range(DIMENSION)
        )
    )


def commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(
        [
            sp.simplify(
                sum(
                    left[index] * sp.diff(right[component], COORDINATES[index])
                    - right[index]
                    * sp.diff(left[component], COORDINATES[index])
                    for index in range(DIMENSION)
                )
            )
            for component in range(DIMENSION)
        ]
    )


@dataclass(frozen=True)
class Geometry:
    l: sp.Matrix
    n: sp.Matrix
    m: sp.Matrix
    mbar: sp.Matrix
    inverse_metric: sp.Matrix
    metric: sp.Matrix
    rho: sp.Expr
    epsilon: sp.Expr
    alpha: sp.Expr
    beta: sp.Expr


def repository_schwarzschild_geometry() -> Geometry:
    # Ripley et al. Eq. (5), with M=1, L=2, a=0.
    l = sp.Matrix(
        [R**2 / 4, R**2 * (R - 2) / 16, sp.Integer(0), sp.Integer(0)]
    )
    n = sp.Matrix([R + 2, R**2 / 4, sp.Integer(0), sp.Integer(0)])
    m = sp.Matrix(
        [
            sp.Integer(0),
            sp.Integer(0),
            -SQRT_TWO * R / 8,
            -SQRT_TWO * I * R / (8 * sp.sin(THETA)),
        ]
    )
    mbar = sp.conjugate(m)
    inverse_metric = sp.simplify(
        l * n.T + n * l.T - m * mbar.T - mbar * m.T
    )
    metric = sp.simplify(inverse_metric.inv())

    # Derive the nonzero real Schwarzschild connection coefficients from the
    # exact tetrad commutators, independently of Ripley et al. Eq. (8).
    tetrad = sp.Matrix.hstack(l, n, m, mbar)
    tetrad_inverse = sp.simplify(tetrad.inv())

    def decompose(vector: sp.Matrix) -> sp.Matrix:
        return sp.simplify(tetrad_inverse * vector)

    delta_d = decompose(commutator(n, l))
    delta_bar_delta = decompose(commutator(mbar, m))
    delta_d_small = decompose(commutator(m, l))

    epsilon = sp.simplify(delta_d[1] / 2)
    rho = sp.simplify(-delta_d_small[2])
    alpha_plus_beta = sp.simplify(delta_d_small[0])
    alpha_minus_beta = sp.simplify(delta_bar_delta[2])
    alpha = sp.simplify((alpha_plus_beta + alpha_minus_beta) / 2)
    beta = sp.simplify((alpha_plus_beta - alpha_minus_beta) / 2)

    require_symbolic_zero("code tetrad l dot n normalization",
                          (l.T * metric * n)[0] - 1)
    require_symbolic_zero("code tetrad m dot mbar normalization",
                          (m.T * metric * mbar)[0] + 1)
    require_symbolic_vector_zero(
        "code tetrad remaining null orthogonality",
        sp.Matrix(
            [
                (l.T * metric * l)[0],
                (n.T * metric * n)[0],
                (m.T * metric * m)[0],
                (l.T * metric * m)[0],
                (l.T * metric * mbar)[0],
                (n.T * metric * m)[0],
                (n.T * metric * mbar)[0],
            ]
        ),
    )
    require_symbolic_zero("Schwarzschild alpha plus beta commutator",
                          alpha + beta)
    require_symbolic_vector_zero(
        "Schwarzschild tau and pi commutator channels",
        sp.Matrix([delta_d[2], delta_d[3]]),
    )
    return Geometry(l, n, m, mbar, inverse_metric, metric, rho, epsilon,
                    alpha, beta)


def reconstruct_org_metric(
    geometry: Geometry,
    h_ll: sp.Expr,
    h_lm: sp.Expr,
    h_mm: sp.Expr,
) -> sp.Matrix:
    n_covector = geometry.metric * geometry.n
    m_covector = geometry.metric * geometry.m
    mbar_covector = geometry.metric * geometry.mbar
    h_lmbar = sp.conjugate(h_lm)
    h_mbar_mbar = sp.conjugate(h_mm)
    return sp.simplify(
        h_ll * (n_covector * n_covector.T)
        - h_lm * (n_covector * mbar_covector.T
                  + mbar_covector * n_covector.T)
        - h_lmbar * (n_covector * m_covector.T
                     + m_covector * n_covector.T)
        + h_mm * (mbar_covector * mbar_covector.T)
        + h_mbar_mbar * (m_covector * m_covector.T)
    )


def checked_org_metric(
    geometry: Geometry,
    fixture_name: str,
    h_ll: sp.Expr,
    h_lm: sp.Expr,
    h_mm: sp.Expr,
) -> sp.Matrix:
    perturbation = reconstruct_org_metric(geometry, h_ll, h_lm, h_mm)
    require_symbolic_zero(
        f"{fixture_name} ORG reconstructed h_ll projection",
        (geometry.l.T * perturbation * geometry.l)[0] - h_ll,
    )
    require_symbolic_zero(
        f"{fixture_name} ORG reconstructed h_lm projection",
        (geometry.l.T * perturbation * geometry.m)[0] - h_lm,
    )
    require_symbolic_zero(
        f"{fixture_name} ORG reconstructed h_mm projection",
        (geometry.m.T * perturbation * geometry.m)[0] - h_mm,
    )
    require_symbolic_vector_zero(
        f"{fixture_name} ORG reconstructed n contraction",
        perturbation * geometry.n,
    )
    require_symbolic_zero(
        f"{fixture_name} ORG reconstructed trace",
        sum(
            geometry.inverse_metric[row, column] * perturbation[row, column]
            for row in range(DIMENSION)
            for column in range(DIMENSION)
        ),
    )
    require_symbolic_vector_zero(
        f"{fixture_name} reconstructed coordinate metric is real",
        sp.Matrix(
            [
                perturbation[row, column]
                - sp.conjugate(perturbation[row, column])
                for row in range(DIMENSION)
                for column in range(DIMENSION)
            ]
        ),
    )
    return perturbation


def corrected_np_psi0(
    geometry: Geometry,
    h_ll: sp.Expr,
    h_lm: sp.Expr,
    h_mm: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    # Schwarzschild has real rho, epsilon, alpha, beta and tau=pi=0.
    sigma = sp.expand(sp.Rational(1, 2) * directional(geometry.l, h_mm))
    kappa = sp.expand(
        directional(geometry.l, h_lm)
        - (2 * geometry.epsilon + geometry.rho) * h_lm
        - sp.Rational(1, 2)
        * (
            directional(geometry.m, h_ll)
            - 2 * geometry.alpha * h_ll
            - 2 * geometry.beta * h_ll
        )
    )
    radial = (
        directional(geometry.l, sigma)
        - (2 * geometry.rho + 2 * geometry.epsilon) * sigma
    )
    corrected = sp.expand(
        radial
        - (
            directional(geometry.m, kappa)
            - geometry.alpha * kappa
            - 3 * geometry.beta * kappa
        )
    )
    old_displayed = sp.expand(
        radial
        - (
            directional(geometry.m, kappa)
            + geometry.alpha * kappa
            + 3 * geometry.beta * kappa
        )
    )
    return corrected, old_displayed


def evaluate(expression: sp.Expr, point: dict[sp.Symbol, sp.Expr]) -> complex:
    return complex(sp.N(expression.subs(point), 18))


def matrix_jets(
    matrix: sp.Matrix, point: dict[sp.Symbol, sp.Expr]
) -> tuple[list[list[complex]], list[list[list[complex]]],
           list[list[list[list[complex]]]]]:
    value = [
        [evaluate(matrix[row, column], point) for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    first = [
        [
            [
                evaluate(sp.diff(matrix[row, column], COORDINATES[derivative]),
                         point)
                for column in range(DIMENSION)
            ]
            for row in range(DIMENSION)
        ]
        for derivative in range(DIMENSION)
    ]
    second = [
        [
            [
                [
                    evaluate(
                        sp.diff(
                            matrix[row, column],
                            COORDINATES[first_derivative],
                            COORDINATES[second_derivative],
                        ),
                        point,
                    )
                    for column in range(DIMENSION)
                ]
                for row in range(DIMENSION)
            ]
            for second_derivative in range(DIMENSION)
        ]
        for first_derivative in range(DIMENSION)
    ]
    return value, first, second


def coordinate_linearized_psi0(
    geometry: Geometry,
    perturbation: sp.Matrix,
    point: dict[sp.Symbol, sp.Expr],
) -> tuple[complex, float]:
    # This is a direct coordinate variation of
    #   Gamma[g+eta h] and Riemann[g+eta h]
    # evaluated at eta=0.  It shares no sigma/kappa expressions with the NP
    # route.  Array order is R^a_{bcd} with
    # R^a_{bcd}=partial_c Gamma^a_db-partial_d Gamma^a_cb+...
    g, dg, ddg = matrix_jets(geometry.metric, point)
    h, dh, ddh = matrix_jets(perturbation, point)
    inverse_g = [
        [evaluate(geometry.inverse_metric[row, column], point)
         for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    derivative_inverse_g = [
        [
            [
                evaluate(
                    sp.diff(geometry.inverse_metric[row, column],
                            COORDINATES[derivative]),
                    point,
                )
                for column in range(DIMENSION)
            ]
            for row in range(DIMENSION)
        ]
        for derivative in range(DIMENSION)
    ]

    metric_sum = [
        [
            [
                dg[b][d][c] + dg[c][d][b] - dg[d][b][c]
                for c in range(DIMENSION)
            ]
            for b in range(DIMENSION)
        ]
        for d in range(DIMENSION)
    ]
    derivative_metric_sum = [
        [
            [
                [
                    ddg[e][b][d][c] + ddg[e][c][d][b]
                    - ddg[e][d][b][c]
                    for c in range(DIMENSION)
                ]
                for b in range(DIMENSION)
            ]
            for d in range(DIMENSION)
        ]
        for e in range(DIMENSION)
    ]
    perturbation_sum = [
        [
            [
                dh[b][d][c] + dh[c][d][b] - dh[d][b][c]
                for c in range(DIMENSION)
            ]
            for b in range(DIMENSION)
        ]
        for d in range(DIMENSION)
    ]
    derivative_perturbation_sum = [
        [
            [
                [
                    ddh[e][b][d][c] + ddh[e][c][d][b]
                    - ddh[e][d][b][c]
                    for c in range(DIMENSION)
                ]
                for b in range(DIMENSION)
            ]
            for d in range(DIMENSION)
        ]
        for e in range(DIMENSION)
    ]

    christoffel = [
        [
            [
                0.5
                * sum(
                    inverse_g[a][d] * metric_sum[d][b][c]
                    for d in range(DIMENSION)
                )
                for c in range(DIMENSION)
            ]
            for b in range(DIMENSION)
        ]
        for a in range(DIMENSION)
    ]
    derivative_christoffel = [
        [
            [
                [
                    0.5
                    * sum(
                        derivative_inverse_g[e][a][d]
                        * metric_sum[d][b][c]
                        + inverse_g[a][d]
                        * derivative_metric_sum[e][d][b][c]
                        for d in range(DIMENSION)
                    )
                    for c in range(DIMENSION)
                ]
                for b in range(DIMENSION)
            ]
            for a in range(DIMENSION)
        ]
        for e in range(DIMENSION)
    ]

    delta_inverse_g = [
        [
            -sum(
                inverse_g[a][p] * h[p][q] * inverse_g[q][d]
                for p in range(DIMENSION)
                for q in range(DIMENSION)
            )
            for d in range(DIMENSION)
        ]
        for a in range(DIMENSION)
    ]
    derivative_delta_inverse_g = [
        [
            [
                -sum(
                    derivative_inverse_g[e][a][p] * h[p][q]
                    * inverse_g[q][d]
                    + inverse_g[a][p] * dh[e][p][q] * inverse_g[q][d]
                    + inverse_g[a][p] * h[p][q]
                    * derivative_inverse_g[e][q][d]
                    for p in range(DIMENSION)
                    for q in range(DIMENSION)
                )
                for d in range(DIMENSION)
            ]
            for a in range(DIMENSION)
        ]
        for e in range(DIMENSION)
    ]
    delta_christoffel = [
        [
            [
                0.5
                * sum(
                    delta_inverse_g[a][d] * metric_sum[d][b][c]
                    + inverse_g[a][d] * perturbation_sum[d][b][c]
                    for d in range(DIMENSION)
                )
                for c in range(DIMENSION)
            ]
            for b in range(DIMENSION)
        ]
        for a in range(DIMENSION)
    ]
    derivative_delta_christoffel = [
        [
            [
                [
                    0.5
                    * sum(
                        derivative_delta_inverse_g[e][a][d]
                        * metric_sum[d][b][c]
                        + delta_inverse_g[a][d]
                        * derivative_metric_sum[e][d][b][c]
                        + derivative_inverse_g[e][a][d]
                        * perturbation_sum[d][b][c]
                        + inverse_g[a][d]
                        * derivative_perturbation_sum[e][d][b][c]
                        for d in range(DIMENSION)
                    )
                    for c in range(DIMENSION)
                ]
                for b in range(DIMENSION)
            ]
            for a in range(DIMENSION)
        ]
        for e in range(DIMENSION)
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

    # Verify that the independently reconstructed background is vacuum.
    maximum_ricci = 0.0
    for b in range(DIMENSION):
        for d in range(DIMENSION):
            ricci = sum(riemann_up(a, b, a, d) for a in range(DIMENSION))
            maximum_ricci = max(maximum_ricci, abs(ricci))

    l = [evaluate(component, point) for component in geometry.l]
    m = [evaluate(component, point) for component in geometry.m]
    contraction = 0.0j
    for a, b, c, d in itertools.product(range(DIMENSION), repeat=4):
        delta_riemann_down = sum(
            h[a][e] * riemann_up(e, b, c, d)
            + g[a][e] * delta_riemann_up(e, b, c, d)
            for e in range(DIMENSION)
        )
        contraction += l[a] * m[b] * l[c] * m[d] * delta_riemann_down

    # All Weyl trace-subtraction terms vanish in the l,m,l,m contraction
    # because l.l=l.m=m.m=0.  The repository convention is Psi0=-C(l,m,l,m).
    return -contraction, maximum_ricci


def manufactured_fixtures() -> list[
    tuple[str, sp.Expr, sp.Expr, sp.Expr, dict[sp.Symbol, sp.Expr]]
]:
    axisymmetric_h_ll = (
        1 + T / 5 + R / 7 + T * R / 11
    ) * (1 + sp.cos(THETA) / 13)
    axisymmetric_h_lm = (
        sp.Rational(2, 7) + T / 9 - R / 8 + R**2 / 10
    ) * sp.sin(THETA)
    axisymmetric_h_mm = (
        sp.Rational(1, 3) - T / 6 + T**2 / 12 + R * T / 14
    ) * sp.sin(THETA) ** 2

    helical_h_ll = (
        sp.Rational(4, 5) + T / 8 - R / 6 + T * R / 15
    ) * (1 + sp.sin(THETA) * sp.cos(PHI) / 9)
    helical_h_lm = (
        sp.Rational(3, 11) - T / 10 + R / 12 + T**2 / 17
    ) * sp.sin(THETA) * sp.exp(I * PHI)
    helical_h_mm = (
        sp.Rational(5, 13) + T / 7 - R * T / 19 + R**2 / 23
    ) * sp.sin(THETA) ** 2 * sp.exp(2 * I * PHI)

    return [
        (
            "axisymmetric",
            axisymmetric_h_ll,
            axisymmetric_h_lm,
            axisymmetric_h_mm,
            {T: sp.Rational(2, 9), R: sp.Rational(7, 10),
             THETA: sp.Rational(4, 5), PHI: sp.Rational(1, 3)},
        ),
        (
            "helical",
            helical_h_ll,
            helical_h_lm,
            helical_h_mm,
            {T: sp.Rational(-1, 6), R: sp.Rational(6, 5),
             THETA: sp.Rational(11, 10), PHI: sp.Rational(3, 7)},
        ),
    ]


def main() -> None:
    geometry = repository_schwarzschild_geometry()
    for name, h_ll, h_lm, h_mm, point in manufactured_fixtures():
        perturbation = checked_org_metric(geometry, name, h_ll, h_lm, h_mm)
        coordinate_value, maximum_ricci = coordinate_linearized_psi0(
            geometry, perturbation, point
        )
        corrected_expression, old_expression = corrected_np_psi0(
            geometry, h_ll, h_lm, h_mm
        )
        corrected_value = evaluate(corrected_expression, point)
        old_value = evaluate(old_expression, point)
        require_close(f"{name} background coordinate Ricci tensor", maximum_ricci,
                      0.0, 2.0e-13)
        require_close(
            f"{name} coordinate Weyl equals corrected NP T0",
            coordinate_value,
            corrected_value,
        )
        require_separated(
            f"{name} coordinate Weyl rejects displayed Eq12 signs",
            coordinate_value,
            old_value,
        )

    print("Completed independent linear Psi0 coordinate-oracle checks")


if __name__ == "__main__":
    main()
