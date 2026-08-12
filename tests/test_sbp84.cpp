#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "teuk/sbp84.hpp"
#include "teuk/types.hpp"

namespace {

double smooth_value(const double x) {
  return std::sin(1.7 * x) + 0.2 * std::cos(2.3 * x);
}

double smooth_derivative(const double x) {
  return 1.7 * std::cos(1.7 * x) - 0.46 * std::sin(2.3 * x);
}

double maximum_error(const std::size_t points) {
  const teuk::UniformRadialGrid grid(points, -0.1, 1.2);
  std::vector<double> values(points);
  std::vector<double> derivative(points);
  for (std::size_t i = 0; i < points; ++i) {
    values[i] = smooth_value(grid.coordinate(i));
  }
  teuk::d84_first_derivative(grid, values, derivative);
  double error = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    error = std::max(
        error,
        std::abs(derivative[i] - smooth_derivative(grid.coordinate(i))));
  }
  return error;
}

}  // namespace

TEST_CASE("D8-4 diagonal norm and derivative satisfy the SBP identity") {
  for (const std::size_t points : {16U, 17U, 25U, 40U}) {
    const teuk::UniformRadialGrid grid(points, 0.0, 1.4);
    double residual = 0.0;
    for (std::size_t i = 0; i < points; ++i) {
      CHECK(teuk::d84_norm_matrix_entry(grid, i, i) > 0.0);
      for (std::size_t j = 0; j < points; ++j) {
        const double hd =
            teuk::d84_norm_matrix_entry(grid, i, i) *
                teuk::d84_derivative_matrix_entry(grid, i, j) +
            teuk::d84_derivative_matrix_entry(grid, j, i) *
                teuk::d84_norm_matrix_entry(grid, j, j);
        const double boundary =
            i == j && i == 0
                ? -1.0
                : (i == j && i + 1 == points ? 1.0 : 0.0);
        residual = std::max(residual, std::abs(hd - boundary));
      }
    }
    CHECK(residual < 2.0e-13);
  }
}

TEST_CASE("D8-4 is quartic-exact globally and degree-eight exact inside") {
  const teuk::UniformRadialGrid grid(33, -0.25, 1.1);
  std::vector<double> values(grid.size());
  std::vector<double> derivative(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = 0.3 - 0.7 * x + 0.4 * x * x - 0.2 * x * x * x +
                0.1 * x * x * x * x;
  }
  teuk::d84_first_derivative(grid, values, derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(derivative[i],
               -0.7 + 0.8 * x - 0.6 * x * x + 0.4 * x * x * x,
               3.0e-11);
  }

  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = std::pow(x, 8);
  }
  teuk::d84_first_derivative(grid, values, derivative);
  for (std::size_t i = teuk::d84_boundary_width;
       i + teuk::d84_boundary_width < grid.size(); ++i) {
    CHECK_NEAR(derivative[i], 8.0 * std::pow(grid.coordinate(i), 7),
               2.0e-11);
  }
}

TEST_CASE("D8-4 has fourth-order maximum-norm boundary convergence") {
  const double coarse = maximum_error(33);
  const double medium = maximum_error(65);
  const double fine = maximum_error(129);
  CHECK(coarse / medium > 13.0);
  CHECK(medium / fine > 14.0);
}

TEST_CASE("D8-4 compatible dissipation is SBP-negative and quartic exact") {
  const teuk::UniformRadialGrid grid(29, -0.2, 1.0);
  std::vector<teuk::Complex> values(grid.size());
  std::vector<teuk::Complex> dissipation(grid.size());
  constexpr double strength = 0.11;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = teuk::Complex(
        std::sin(8.0 * x) + 0.3 * std::cos(5.0 * x),
        0.4 * std::sin(3.0 * x));
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    dissipation[i] = teuk::d84_compatible_dissipation_at(
        values.data(), grid.size(), i, grid.spacing(), strength);
  }
  teuk::Complex energy = 0.0;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    energy += grid.spacing() * teuk::d84_norm_weight(grid.size(), i) *
              Kokkos::conj(values[i]) * dissipation[i];
  }
  double expected = 0.0;
  for (std::size_t i = 0; i + 5 < grid.size(); ++i) {
    const teuk::Complex difference =
        -values[i] + 5.0 * values[i + 1] - 10.0 * values[i + 2] +
        10.0 * values[i + 3] - 5.0 * values[i + 4] + values[i + 5];
    expected -= strength * Kokkos::abs(difference) * Kokkos::abs(difference);
  }
  CHECK(energy.real() <= 0.0);
  CHECK_NEAR(energy.real(), expected, 3.0e-12);
  CHECK_NEAR(energy.imag(), 0.0, 3.0e-12);

  std::vector<double> quartic(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    quartic[i] = 0.2 - x + 0.3 * x * x - 0.4 * x * x * x +
                 0.7 * x * x * x * x;
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK(std::abs(teuk::d84_compatible_dissipation_at(
              quartic.data(), grid.size(), i, grid.spacing(), strength)) <
          2.0e-9);
  }
}

TEST_CASE("D8-4 point derivative and dissipation execute on device") {
  const teuk::UniformRadialGrid grid(20, 0.0, 1.0);
  Kokkos::View<double*> values("d84_values", grid.size());
  Kokkos::View<double*> derivative("d84_derivative", grid.size());
  Kokkos::View<double*> dissipation("d84_dissipation", grid.size());
  auto host = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    host(i) = 0.5 - 0.3 * x + 0.2 * x * x * x * x;
  }
  Kokkos::deep_copy(values, host);
  Kokkos::parallel_for(
      "d84_device_point", grid.size(), KOKKOS_LAMBDA(const int i) {
        const std::size_t index = static_cast<std::size_t>(i);
        derivative(index) = teuk::d84_first_derivative_at(
            values.data(), grid.size(), index, 1.0 / grid.spacing());
        dissipation(index) = teuk::d84_compatible_dissipation_at(
            values.data(), grid.size(), index, grid.spacing(), 0.1);
      });
  const auto host_derivative = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), derivative);
  const auto host_dissipation = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), dissipation);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(host_derivative(i), -0.3 + 0.8 * x * x * x, 4.0e-11);
    CHECK(std::abs(host_dissipation(i)) < 2.0e-9);
  }
}
