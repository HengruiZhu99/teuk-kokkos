#include "test_harness.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/radial_discretization.hpp"

TEST_CASE("radial discretization names and minimum extents are strict") {
  CHECK(std::string(teuk::radial_discretization_name(
            teuk::RadialDiscretization::D42)) == "d4-2");
  CHECK(std::string(teuk::radial_discretization_name(
            teuk::RadialDiscretization::D84)) == "d8-4");
  CHECK(teuk::parse_radial_discretization("d4-2") ==
        teuk::RadialDiscretization::D42);
  CHECK(teuk::parse_radial_discretization("d8-4") ==
        teuk::RadialDiscretization::D84);
  CHECK(teuk::radial_minimum_points(teuk::RadialDiscretization::D42) == 8);
  CHECK(teuk::radial_minimum_points(teuk::RadialDiscretization::D84) == 16);
  bool rejected = false;
  try {
    (void)teuk::parse_radial_discretization("fourth-ish");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("radial dispatch preserves each SBP operator on device") {
  constexpr std::size_t points = 20;
  const teuk::UniformRadialGrid grid(points, -0.2, 1.0);
  Kokkos::View<double*> values("radial_dispatch_values", points);
  Kokkos::View<double**> derivative("radial_dispatch_derivative", 2, points);
  auto host_values = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < points; ++i) {
    const double x = grid.coordinate(i);
    host_values(i) = 0.3 - 0.8 * x + 0.7 * x * x;
  }
  Kokkos::deep_copy(values, host_values);
  Kokkos::parallel_for(
      "radial_dispatch_device", 2 * points,
      KOKKOS_LAMBDA(const int flat) {
        const std::size_t scheme = static_cast<std::size_t>(flat) / points;
        const std::size_t i = static_cast<std::size_t>(flat) - scheme * points;
        const auto discretization =
            scheme == 0 ? teuk::RadialDiscretization::D42
                        : teuk::RadialDiscretization::D84;
        derivative(scheme, i) = teuk::radial_first_derivative_at(
            discretization, values.data(), points, i, 1.0 / grid.spacing());
      });
  const auto host_derivative = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), derivative);
  for (std::size_t scheme = 0; scheme < 2; ++scheme) {
    for (std::size_t i = 0; i < points; ++i) {
      CHECK_NEAR(host_derivative(scheme, i),
                 -0.8 + 1.4 * grid.coordinate(i), 4.0e-11);
    }
  }
}

TEST_CASE("D8-4 dispatch has fourth-order boundary accuracy") {
  const auto error = [](const std::size_t points) {
    const teuk::UniformRadialGrid grid(points, 0.0, 1.0);
    std::vector<double> values(points);
    for (std::size_t i = 0; i < points; ++i) {
      values[i] = std::exp(grid.coordinate(i));
    }
    const double approximate = teuk::radial_first_derivative_at(
        teuk::RadialDiscretization::D84, values.data(), points, 0,
        1.0 / grid.spacing());
    return std::abs(approximate - 1.0);
  };
  const double coarse = error(33);
  const double medium = error(65);
  const double fine = error(129);
  CHECK(coarse / medium > 13.0);
  CHECK(medium / fine > 14.0);
}
