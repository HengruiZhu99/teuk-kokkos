#include <cmath>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"
#include "teuk/teukolsky.hpp"

namespace angular = teuk::angular;

TEST_CASE("Gauss-Legendre quadrature has deterministic symmetry and exact moments") {
  constexpr int node_count = 8;
  const auto grid = angular::gauss_legendre(node_count);
  CHECK(grid.size() == node_count);

  for (int i = 0; i < node_count; ++i) {
    CHECK_NEAR(grid.x[static_cast<std::size_t>(i)],
               -grid.x[static_cast<std::size_t>(node_count - 1 - i)], 2e-15);
    CHECK_NEAR(grid.weights[static_cast<std::size_t>(i)],
               grid.weights[static_cast<std::size_t>(node_count - 1 - i)],
               2e-15);
    CHECK(grid.weights[static_cast<std::size_t>(i)] > 0.0);
    if (i + 1 < node_count) {
      CHECK(grid.x[static_cast<std::size_t>(i)] <
            grid.x[static_cast<std::size_t>(i + 1)]);
    }
  }

  // N-point Gauss-Legendre integrates powers through 2N-1 exactly.
  for (int power = 0; power <= 2 * node_count - 1; ++power) {
    double integral = 0.0;
    for (std::size_t i = 0; i < grid.size(); ++i) {
      integral += grid.weights[i] * std::pow(grid.x[i], power);
    }
    const double exact =
        power % 2 == 0 ? 2.0 / static_cast<double>(power + 1) : 0.0;
    CHECK_NEAR(integral, exact, 3e-14);
  }
}

TEST_CASE("raising lowering and Teukolsky angular factors follow conventions") {
  CHECK_NEAR(angular::raising_factor(2, -2), 2.0, 1e-15);
  CHECK_NEAR(angular::raising_factor(3, -1), std::sqrt(12.0), 1e-15);
  CHECK_NEAR(angular::lowering_factor(2, -2), 0.0, 1e-15);
  CHECK_NEAR(angular::lowering_factor(3, -1), -std::sqrt(10.0), 1e-15);

  for (int ell = 3; ell <= 7; ++ell) {
    for (int spin = -3; spin <= 3; ++spin) {
      if (spin < ell) {
        const double composed = angular::raising_factor(ell, spin) *
                                angular::lowering_factor(ell, spin + 1);
        CHECK_NEAR(composed,
                   angular::lower_after_raise_eigenvalue(ell, spin), 2e-14);
      }
      const double symmetric =
          0.5 * (angular::lower_after_raise_eigenvalue(ell, spin) +
                 angular::raise_after_lower_eigenvalue(ell, spin));
      CHECK_NEAR(symmetric,
                 angular::symmetric_spin_covariant_laplacian_eigenvalue(
                     ell, spin),
                 2e-14);
      CHECK_NEAR(angular::spin_weighted_laplacian_eigenvalue(ell, spin),
                 angular::lower_after_raise_eigenvalue(ell, spin), 2e-14);
    }
  }
}

TEST_CASE("Teukolsky angular eigenvalues reject the former symmetric shift") {
  CHECK_NEAR(angular::spin_weighted_laplacian_eigenvalue(2, -2), -4.0,
             0.0);
  CHECK_NEAR(angular::spin_weighted_laplacian_eigenvalue(3, -2), -10.0,
             0.0);
  CHECK_NEAR(angular::spin_weighted_laplacian_eigenvalue(4, -2), -18.0,
             0.0);
  for (int spin : {-2, -1, 1, 2}) {
    for (int ell = std::abs(spin); ell <= std::abs(spin) + 4; ++ell) {
      const double teukolsky =
          angular::spin_weighted_laplacian_eigenvalue(ell, spin);
      const double old_symmetric =
          angular::symmetric_spin_covariant_laplacian_eigenvalue(ell, spin);
      CHECK_NEAR(old_symmetric - teukolsky, -static_cast<double>(spin),
                 2e-14);
    }
  }

  // This is an explicit L_{s+1} R_s composition for the production spin.
  for (int ell = 2; ell <= 7; ++ell) {
    const double explicit_composition =
        angular::raising_factor(ell, -2) *
        angular::lowering_factor(ell, -1);
    CHECK_NEAR(angular::spin_weighted_laplacian_eigenvalue(ell, -2),
               explicit_composition, 2e-14);
  }
}

TEST_CASE("spin-weighted harmonics reproduce analytic low modes and poles") {
  const double theta = 0.73;
  const double x = std::cos(theta);
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(0, 0, 0, theta),
             1.0 / std::sqrt(4.0 * angular::pi), 2e-15);
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(1, 0, 0, theta),
             std::sqrt(3.0 / (4.0 * angular::pi)) * x, 2e-15);
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(1, 1, 0, theta),
             -std::sqrt(3.0 / (8.0 * angular::pi)) * std::sin(theta),
             2e-15);

  const double normalization = std::sqrt(5.0 / (64.0 * angular::pi));
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(2, 2, -2, theta),
             normalization * std::pow(1.0 + x, 2), 3e-15);
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(2, 2, -2, 0.0),
             4.0 * normalization, 3e-15);
  CHECK_NEAR(angular::spin_weighted_harmonic_theta(2, 2, -2, angular::pi),
             0.0, 3e-15);
}

TEST_CASE("spin-weighted collocation basis is orthonormal") {
  const angular::SpinWeightedTransform transform(-2, 2, 8, 14);
  const auto& synthesis = transform.synthesis_matrix();
  const auto& analysis = transform.analysis_matrix();
  for (std::size_t row = 0; row < transform.mode_count(); ++row) {
    for (std::size_t column = 0; column < transform.mode_count(); ++column) {
      double product = 0.0;
      for (std::size_t node = 0; node < transform.grid().size(); ++node) {
        product += analysis(row, node) * synthesis(node, column);
      }
      CHECK_NEAR(product, row == column ? 1.0 : 0.0, 2e-12);
    }
  }
}

TEST_CASE("spin-weighted modal synthesis and analysis round trip") {
  const angular::SpinWeightedTransform transform(-1, -2, 7, 12);
  std::vector<teuk::Complex> modal(transform.mode_count());
  for (std::size_t i = 0; i < modal.size(); ++i) {
    modal[i] = teuk::Complex(0.25 + static_cast<double>(i),
                            -0.5 + 0.125 * static_cast<double>(i));
  }
  const auto recovered = transform.analyze(transform.synthesize(modal));
  for (std::size_t i = 0; i < modal.size(); ++i) {
    CHECK_COMPLEX_NEAR(recovered[i], modal[i], 3e-12);
  }
}

TEST_CASE("modal angular operators apply exact diagonal eigenvalues") {
  const angular::SpinWeightedTransform transform(-2, 1, 5, 9);
  const std::vector<teuk::Complex> modal = {
      {1.0, -0.5}, {0.25, 2.0}, {-1.5, 0.75}, {2.0, 1.0}};
  const auto raised = transform.raise(modal);
  const auto lowered = transform.lower(modal);
  const auto laplacian = transform.laplacian(modal);
  for (std::size_t i = 0; i < modal.size(); ++i) {
    const int ell = transform.ell_min() + static_cast<int>(i);
    CHECK_COMPLEX_NEAR(
        raised[i], angular::raising_factor(ell, transform.spin()) * modal[i],
        2e-15);
    CHECK_COMPLEX_NEAR(
        lowered[i], angular::lowering_factor(ell, transform.spin()) * modal[i],
        2e-15);
    CHECK_COMPLEX_NEAR(
        laplacian[i],
        angular::spin_weighted_laplacian_eigenvalue(ell, transform.spin()) *
            modal[i],
        2e-15);
  }
}

TEST_CASE("Schwarzschild scri pure Y22 sees the separated angular eigenvalue") {
  const angular::SpinWeightedTransform transform(-2, 2, 4, 7);
  std::vector<teuk::Complex> modal(transform.mode_count(),
                                   teuk::Complex(0.0, 0.0));
  modal[0] = teuk::Complex(0.7, -0.3);
  const auto psi = transform.synthesize(modal);
  const auto angular_rhs = transform.synthesize(transform.laplacian(modal));

  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.0;
  parameters.compactification_length = 1.3;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = 2;
  for (std::size_t node = 0; node < psi.size(); ++node) {
    const auto coefficients = teuk::teukolsky_coefficients(
        parameters, 0.0, transform.grid().theta(node));
    const teuk::TeukolskyState state{teuk::Complex(0.0, 0.0),
                                     teuk::Complex(0.0, 0.0), psi[node]};
    const auto p_rhs = teuk::teukolsky_p_rhs(
        coefficients, state, teuk::Complex(0.0, 0.0), angular_rhs[node],
        teuk::Complex(0.0, 0.0));
    CHECK_COMPLEX_NEAR(p_rhs, -4.0 * psi[node], 3e-14);
  }
}

TEST_CASE("raising and lowering obey the unit-sphere adjoint sign") {
  const angular::SpinWeightedTransform spin_minus_one(-1, 2, 5, 9);
  const angular::SpinWeightedTransform spin_zero(0, 2, 5, 9);
  const std::vector<teuk::Complex> f = {
      {1.0, 0.5}, {-0.25, 2.0}, {0.75, -1.0}, {1.5, 0.25}};
  const std::vector<teuk::Complex> g = {
      {-0.5, 1.0}, {2.0, 0.125}, {1.25, 0.75}, {-1.0, -0.5}};
  const auto raised_f = spin_minus_one.raise(f);
  const auto lowered_g = spin_zero.lower(g);

  teuk::Complex left(0.0, 0.0);
  teuk::Complex right(0.0, 0.0);
  for (std::size_t i = 0; i < f.size(); ++i) {
    left += Kokkos::conj(raised_f[i]) * g[i];
    right += Kokkos::conj(f[i]) * lowered_g[i];
  }
  CHECK_COMPLEX_NEAR(left, -right, 2e-14);
}

TEST_CASE("integer Wigner 3-j oracle obeys exact selection rules") {
  CHECK_NEAR(angular::wigner_3j(0, 0, 0, 0, 0, 0), 1.0, 1e-15);
  for (int ell = 1; ell <= 7; ++ell) {
    for (int m = -ell; m <= ell; ++m) {
      const double expected =
          ((ell - m) & 1) != 0 ? -1.0 / std::sqrt(2.0 * ell + 1.0)
                               : 1.0 / std::sqrt(2.0 * ell + 1.0);
      CHECK_NEAR(angular::wigner_3j(ell, ell, 0, m, -m, 0), expected,
                 2e-14);
    }
  }
  CHECK_NEAR(angular::wigner_3j(2, 2, 2, 1, 1, 1), 0.0, 0.0);
  CHECK_NEAR(angular::wigner_3j(1, 1, 3, 0, 0, 0), 0.0, 0.0);
  CHECK_NEAR(angular::wigner_3j(2, 2, 2, 3, -3, 0), 0.0, 0.0);
}

TEST_CASE("generalized Gaunt oracle enforces mode spin and triangle coupling") {
  CHECK(angular::gaunt_selection_allowed(2, 1, -1, 3, -2, 1, 4, -1,
                                         0));
  CHECK(!angular::gaunt_selection_allowed(2, 1, -1, 3, -2, 1, 4, 0,
                                          0));
  CHECK(!angular::gaunt_selection_allowed(2, 1, -1, 3, -2, 1, 4, -1,
                                          -1));
  CHECK(!angular::gaunt_selection_allowed(1, 0, 0, 1, 0, 0, 3, 0, 0));

  CHECK_NEAR(angular::spin_weighted_gaunt(2, 1, -1, 3, -2, 1, 4,
                                          0, 0),
             0.0, 0.0);
  CHECK_NEAR(angular::spin_weighted_gaunt(1, 0, 0, 1, 0, 0, 1, 0, 0),
             0.0, 2e-15);  // odd scalar parity
}

TEST_CASE("Y00 Gaunt coupling preserves every normalized harmonic") {
  const double inverse_sqrt_four_pi =
      1.0 / std::sqrt(4.0 * angular::pi);
  for (int spin = -2; spin <= 2; ++spin) {
    for (int ell = std::max(2, std::abs(spin)); ell <= 6; ++ell) {
      for (int m = -ell; m <= ell; ++m) {
        CHECK_NEAR(angular::spin_weighted_gaunt(
                       ell, m, spin, 0, 0, 0, ell, m, spin),
                   inverse_sqrt_four_pi, 5e-14);
      }
    }
  }
}

TEST_CASE("padded spin-weighted product agrees with exact Gaunt convolution") {
  const angular::SpinWeightedTransform left(-1, 1, 5, 7);
  const angular::SpinWeightedTransform right(-1, 1, 5, 7);
  const angular::SpinWeightedTransform target(-2, 2, 6, 7);
  const std::vector<teuk::Complex> left_modal = {
      {0.5, -0.25}, {1.0, 0.125}, {-0.75, 0.5}, {0.25, 0.75},
      {-0.125, -0.5}};
  const std::vector<teuk::Complex> right_modal = {
      {-0.25, 0.5}, {0.75, -0.125}, {0.5, 0.25}, {-1.0, 0.375},
      {0.125, 0.5}};

  const auto exact = angular::exact_modal_product(
      left, left_modal, right, right_modal, target);
  const auto padded = angular::collocation_product(
      left, left_modal, right, right_modal, target);
  CHECK(angular::padded_product_node_count(left, right, target) == 9);
  for (std::size_t i = 0; i < exact.size(); ++i) {
    CHECK_COMPLEX_NEAR(padded[i], exact[i], 2e-11);
  }
}

TEST_CASE("unpadded angular product exposes aliasing that padding removes") {
  const angular::SpinWeightedTransform factor(0, 0, 4, 5);
  const angular::SpinWeightedTransform target(0, 0, 4, 5);
  std::vector<teuk::Complex> pure_ell_four(factor.mode_count(),
                                          teuk::Complex(0.0, 0.0));
  pure_ell_four[4] = teuk::Complex(1.0, 0.0);

  const auto exact = angular::exact_modal_product(
      factor, pure_ell_four, factor, pure_ell_four, target);
  const auto unpadded = angular::collocation_product(
      factor, pure_ell_four, factor, pure_ell_four, target, 5);
  const auto padded = angular::collocation_product(
      factor, pure_ell_four, factor, pure_ell_four, target);

  double unpadded_error = 0.0;
  double padded_error = 0.0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    unpadded_error =
        std::max(unpadded_error, static_cast<double>(Kokkos::abs(unpadded[i] - exact[i])));
    padded_error =
        std::max(padded_error, static_cast<double>(Kokkos::abs(padded[i] - exact[i])));
  }
  CHECK(unpadded_error > 1e-3);
  CHECK(padded_error < 2e-12);
}
