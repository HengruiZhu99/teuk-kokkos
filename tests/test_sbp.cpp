#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "teuk/grid.hpp"
#include "teuk/sbp.hpp"

namespace {

double sbp_smooth_value(const double radius) {
  return std::sin(2.1 * radius) + 0.3 * std::cos(1.3 * radius);
}

double sbp_smooth_derivative(const double radius) {
  return 2.1 * std::cos(2.1 * radius) - 0.39 * std::sin(1.3 * radius);
}

struct SbpErrors {
  double boundary;
  double interior;
};

SbpErrors sbp_errors(const std::size_t point_count) {
  const teuk::UniformRadialGrid grid(point_count, 0.0, 1.2);
  std::vector<double> values(point_count);
  std::vector<double> derivative(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    values[i] = sbp_smooth_value(grid.coordinate(i));
  }
  teuk::d42_first_derivative(grid, values, derivative);
  SbpErrors errors{0.0, 0.0};
  for (std::size_t i = 0; i < point_count; ++i) {
    const double error = std::abs(
        derivative[i] - sbp_smooth_derivative(grid.coordinate(i)));
    if (i < 4 || i + 4 >= point_count) {
      errors.boundary = std::max(errors.boundary, error);
    } else {
      errors.interior = std::max(errors.interior, error);
    }
  }
  return errors;
}

}  // namespace

TEST_CASE("D4-2 diagonal norm and derivative satisfy the SBP identity") {
  for (const std::size_t point_count : {8U, 9U, 17U, 32U}) {
    const teuk::UniformRadialGrid grid(point_count, 0.0, 2.3);
    double maximum_residual = 0.0;
    for (std::size_t i = 0; i < point_count; ++i) {
      CHECK(teuk::d42_norm_matrix_entry(grid, i, i) > 0.0);
      for (std::size_t j = 0; j < point_count; ++j) {
        const double h_d =
            teuk::d42_norm_matrix_entry(grid, i, i) *
                teuk::d42_derivative_matrix_entry(grid, i, j) +
            teuk::d42_derivative_matrix_entry(grid, j, i) *
                teuk::d42_norm_matrix_entry(grid, j, j);
        const double boundary =
            i == j && i == 0
                ? -1.0
                : (i == j && i + 1 == point_count ? 1.0 : 0.0);
        maximum_residual =
            std::max(maximum_residual, std::abs(h_d - boundary));
      }
    }
    CHECK(maximum_residual < 3.0e-16);
  }
}

TEST_CASE("D4-2 is quadratic-exact globally and quartic-exact in the interior") {
  const teuk::UniformRadialGrid grid(21, -0.2, 1.3);
  std::vector<double> values(grid.size());
  std::vector<double> derivative(grid.size());

  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = 0.4 - 0.7 * x + 1.2 * x * x;
  }
  teuk::d42_first_derivative(grid, values, derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK_NEAR(derivative[i], -0.7 + 2.4 * grid.coordinate(i), 2.0e-13);
  }

  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = x * x * x * x;
  }
  teuk::d42_first_derivative(grid, values, derivative);
  for (std::size_t i = 4; i + 4 < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(derivative[i], 4.0 * x * x * x, 3.0e-13);
  }
}

TEST_CASE("D4-2 exhibits second-order boundary and fourth-order interior design") {
  const SbpErrors error_33 = sbp_errors(33);
  const SbpErrors error_65 = sbp_errors(65);
  const SbpErrors error_129 = sbp_errors(129);
  CHECK(error_33.boundary / error_65.boundary > 3.5);
  CHECK(error_65.boundary / error_129.boundary > 3.7);
  CHECK(error_33.interior / error_65.interior > 13.0);
  CHECK(error_65.interior / error_129.interior > 14.0);
}

TEST_CASE("D4-2 compatible dissipation is nonpositive in the SBP norm") {
  const teuk::UniformRadialGrid grid(25, 0.0, 1.0);
  std::vector<teuk::Complex> values(grid.size());
  std::vector<teuk::Complex> dissipation(grid.size());
  teuk::D42DissipationWorkspace<teuk::Complex> workspace(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = teuk::Complex(std::sin(11.0 * x) + 0.2 * std::cos(7.0 * x),
                              0.4 * std::sin(5.0 * x));
  }
  constexpr double strength = 0.17;
  teuk::apply_d42_compatible_dissipation(
      grid, values, strength, workspace, dissipation);

  teuk::Complex energy_rate = 0.0;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    energy_rate += grid.spacing() *
                   teuk::d42_norm_weight(grid.size(), i) *
                   Kokkos::conj(values[i]) * dissipation[i];
  }
  double expected = 0.0;
  for (const teuk::Complex difference : workspace.third_difference) {
    const double magnitude = Kokkos::abs(difference);
    expected -= strength * magnitude * magnitude;
  }
  CHECK(energy_rate.real() <= 0.0);
  CHECK_NEAR(energy_rate.real(), expected, 2.0e-14);
  CHECK_NEAR(energy_rate.imag(), 0.0, 2.0e-14);
}

TEST_CASE("D4-2 compatible dissipation annihilates quadratic data") {
  const teuk::UniformRadialGrid grid(16, -0.4, 1.0);
  std::vector<double> values(grid.size());
  std::vector<double> dissipation(grid.size());
  teuk::D42DissipationWorkspace<double> workspace(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = 0.2 - 0.8 * x + 1.7 * x * x;
  }
  teuk::apply_d42_compatible_dissipation(
      grid, values, 0.2, workspace, dissipation);
  for (const double value : dissipation) CHECK(std::abs(value) < 2.0e-11);
}

TEST_CASE("D4-2 point derivative executes through active Kokkos space") {
  const teuk::UniformRadialGrid grid(12, 0.0, 1.0);
  Kokkos::View<double*> values("sbp_values", grid.size());
  Kokkos::View<double*> derivative("sbp_derivative", grid.size());
  auto host_values = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    host_values(i) = 0.3 - x + 0.7 * x * x;
  }
  Kokkos::deep_copy(values, host_values);
  Kokkos::parallel_for(
      "d42_point_derivative", grid.size(), KOKKOS_LAMBDA(const int i) {
        derivative(i) = teuk::d42_first_derivative_at(
            values.data(), grid.size(), static_cast<std::size_t>(i),
            1.0 / grid.spacing());
      });
  const auto host_derivative =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK_NEAR(host_derivative(i), -1.0 + 1.4 * grid.coordinate(i),
               2.0e-13);
  }
}
