#include <cmath>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"

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

TEST_CASE("raising lowering and spin Laplacian factors follow fixed conventions") {
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
                 angular::spin_weighted_laplacian_eigenvalue(ell, spin),
                 2e-14);
    }
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
