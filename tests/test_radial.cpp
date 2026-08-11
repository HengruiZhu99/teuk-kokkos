#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "teuk/grid.hpp"
#include "teuk/radial.hpp"

namespace {

double smooth_value(const double radius) {
  return std::sin(2.3 * radius) + 0.2 * std::cos(1.7 * radius);
}

double smooth_derivative(const double radius) {
  return 2.3 * std::cos(2.3 * radius) - 0.34 * std::sin(1.7 * radius);
}

double derivative_error(const std::size_t points) {
  const teuk::UniformRadialGrid grid(points, 0.0, 1.4);
  std::vector<double> values(points);
  std::vector<double> derivative(points);
  for (std::size_t i = 0; i < points; ++i) {
    values[i] = smooth_value(grid.coordinate(i));
  }
  teuk::fourth_order_radial_derivative(grid, values, derivative);
  double error = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    error = std::max(
        error, std::abs(derivative[i] - smooth_derivative(grid.coordinate(i))));
  }
  return error;
}

}  // namespace

TEST_CASE("uniform radial grid runs from scri to the horizon") {
  const teuk::UniformRadialGrid grid(9, 2.0);
  CHECK(grid.size() == 9);
  CHECK_NEAR(grid.lower_radius(), 0.0, 0.0);
  CHECK_NEAR(grid.upper_radius(), 2.0, 0.0);
  CHECK_NEAR(grid.spacing(), 0.25, 1.0e-15);
  CHECK_NEAR(grid.coordinate(0), 0.0, 0.0);
  CHECK_NEAR(grid.coordinate(8), 2.0, 1.0e-15);
}

TEST_CASE("fourth-order radial closures differentiate quartics exactly") {
  const teuk::UniformRadialGrid grid(13, -0.3, 1.1);
  std::vector<double> values(grid.size());
  std::vector<double> derivative(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = 0.7 - 1.2 * x + 0.4 * x * x - 0.3 * x * x * x +
                0.9 * x * x * x * x;
  }
  teuk::fourth_order_radial_derivative(grid, values, derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    const double exact = -1.2 + 0.8 * x - 0.9 * x * x +
                         3.6 * x * x * x;
    CHECK_NEAR(derivative[i], exact, 3.0e-12);
  }
}

TEST_CASE("radial derivative converges at fourth order including boundaries") {
  const double error_17 = derivative_error(17);
  const double error_33 = derivative_error(33);
  const double error_65 = derivative_error(65);
  CHECK(error_17 / error_33 > 12.0);
  CHECK(error_33 / error_65 > 13.0);
}

TEST_CASE("radial derivative supports complex fields") {
  const teuk::UniformRadialGrid grid(9, 0.0, 1.0);
  std::vector<std::complex<double>> values(grid.size());
  std::vector<std::complex<double>> derivative(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = {x * x * x, -2.0 * x * x};
  }
  teuk::fourth_order_radial_derivative(grid, values, derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK(std::abs(derivative[i] -
                   std::complex<double>(3.0 * x * x, -4.0 * x)) < 2.0e-13);
  }
}

TEST_CASE("point radial stencil executes through the active Kokkos space") {
  const teuk::UniformRadialGrid grid(7, 0.0, 1.0);
  Kokkos::View<double*> values("radial_values", grid.size());
  Kokkos::View<double*> derivative("radial_derivative", grid.size());
  auto host_values = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    host_values(i) = x * x * x * x;
  }
  Kokkos::deep_copy(values, host_values);
  Kokkos::parallel_for(
      "reference_radial_stencil", grid.size(), KOKKOS_LAMBDA(const int i) {
        derivative(i) = teuk::fourth_order_radial_derivative_at(
            values.data(), grid.size(), static_cast<std::size_t>(i),
            1.0 / grid.spacing());
      });
  const auto host_derivative =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(host_derivative(i), 4.0 * x * x * x, 2.0e-12);
  }
}
