#include "test_harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/radial_discretization.hpp"

TEST_CASE("radial discretization names and minimum extents are strict") {
  CHECK(std::string(teuk::radial_discretization_name(
            teuk::RadialDiscretization::D42)) == "d4-2");
  CHECK(std::string(teuk::radial_discretization_name(
            teuk::RadialDiscretization::D84)) == "d8-4");
  CHECK(std::string(teuk::radial_discretization_name(
            teuk::RadialDiscretization::D105)) == "d10-5");
  CHECK(teuk::parse_radial_discretization("d4-2") ==
        teuk::RadialDiscretization::D42);
  CHECK(teuk::parse_radial_discretization("d8-4") ==
        teuk::RadialDiscretization::D84);
  CHECK(teuk::parse_radial_discretization("d10-5") ==
        teuk::RadialDiscretization::D105);
  CHECK(teuk::radial_minimum_points(teuk::RadialDiscretization::D42) == 8);
  CHECK(teuk::radial_minimum_points(teuk::RadialDiscretization::D84) == 16);
  CHECK(teuk::radial_minimum_points(teuk::RadialDiscretization::D105) == 22);
  bool rejected = false;
  try {
    (void)teuk::parse_radial_discretization("fourth-ish");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("radial dissipation timestep helper fails closed") {
  CHECK(!teuk::radial_dissipation_rk4_step_is_admissible(
      teuk::RadialDiscretization::D105, 0.006, -0.1));
  CHECK(!teuk::radial_dissipation_rk4_step_is_admissible(
      teuk::RadialDiscretization::D105, 0.006,
      std::numeric_limits<double>::infinity()));
  CHECK(!teuk::radial_dissipation_rk4_step_is_admissible(
      teuk::RadialDiscretization::D105,
      std::numeric_limits<double>::quiet_NaN(), 0.01));
  CHECK(teuk::radial_dissipation_rk4_step_is_admissible(
      teuk::RadialDiscretization::D105, 0.0, 1.0e6));
}

TEST_CASE("radial dissipation spectral bounds cover all closure patterns") {
  const auto maximum_row_sum = [](const teuk::RadialDiscretization scheme,
                                  const std::size_t points) {
    constexpr std::array<double, 7> d105{1.0, -6.0, 15.0, -20.0,
                                         15.0, -6.0, 1.0};
    constexpr std::array<double, 7> d84{-1.0, 5.0, -10.0, 10.0,
                                        -5.0, 1.0, 0.0};
    constexpr std::array<double, 7> d42{-1.0, 3.0, -3.0, 1.0,
                                        0.0, 0.0, 0.0};
    const auto& stencil = scheme == teuk::RadialDiscretization::D105
                              ? d105
                              : (scheme == teuk::RadialDiscretization::D84
                                     ? d84
                                     : d42);
    const std::size_t width =
        scheme == teuk::RadialDiscretization::D105
            ? 7
            : (scheme == teuk::RadialDiscretization::D84 ? 6 : 4);
    const std::size_t rows = points - width + 1;
    double maximum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
      double row_sum = 0.0;
      for (std::size_t other = 0; other < rows; ++other) {
        double entry = 0.0;
        for (std::size_t position = 0; position < width; ++position) {
          const std::size_t column = row + position;
          if (column >= other && column < other + width) {
            entry += stencil[position] * stencil[column - other] /
                     teuk::radial_norm_weight(scheme, points, column);
          }
        }
        row_sum += std::abs(entry);
      }
      maximum = std::max(maximum, row_sum);
    }
    return maximum;
  };

  for (const auto scheme : {teuk::RadialDiscretization::D42,
                            teuk::RadialDiscretization::D84,
                            teuk::RadialDiscretization::D105}) {
    for (std::size_t points = teuk::radial_minimum_points(scheme);
         points <= 64; ++points) {
      CHECK(maximum_row_sum(scheme, points) <=
            teuk::radial_dissipation_spectral_radius_bound(scheme));
    }
  }
}

TEST_CASE("radial dispatch preserves each SBP operator on device") {
  constexpr std::size_t points = 24;
  const teuk::UniformRadialGrid grid(points, -0.2, 1.0);
  Kokkos::View<double*> values("radial_dispatch_values", points);
  Kokkos::View<double**> derivative("radial_dispatch_derivative", 3, points);
  auto host_values = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < points; ++i) {
    const double x = grid.coordinate(i);
    host_values(i) = 0.3 - 0.8 * x + 0.7 * x * x;
  }
  Kokkos::deep_copy(values, host_values);
  Kokkos::parallel_for(
      "radial_dispatch_device", 3 * points,
      KOKKOS_LAMBDA(const int flat) {
        const std::size_t scheme = static_cast<std::size_t>(flat) / points;
        const std::size_t i = static_cast<std::size_t>(flat) - scheme * points;
        const auto discretization =
            scheme == 0   ? teuk::RadialDiscretization::D42
            : scheme == 1 ? teuk::RadialDiscretization::D84
                          : teuk::RadialDiscretization::D105;
        derivative(scheme, i) = teuk::radial_first_derivative_at(
            discretization, values.data(), points, i, 1.0 / grid.spacing());
      });
  const auto host_derivative = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), derivative);
  for (std::size_t scheme = 0; scheme < 3; ++scheme) {
    for (std::size_t i = 0; i < points; ++i) {
      CHECK_NEAR(host_derivative(scheme, i),
                 -0.8 + 1.4 * grid.coordinate(i), 4.0e-11);
    }
  }
}

TEST_CASE("D10-5 dispatch has fifth-order boundary accuracy") {
  const auto error = [](const std::size_t points) {
    const teuk::UniformRadialGrid grid(points, 0.0, 1.0);
    std::vector<double> values(points);
    for (std::size_t i = 0; i < points; ++i) {
      values[i] = std::exp(grid.coordinate(i));
    }
    const double approximate = teuk::radial_first_derivative_at(
        teuk::RadialDiscretization::D105, values.data(), points, 0,
        1.0 / grid.spacing());
    return std::abs(approximate - 1.0);
  };
  const double coarse = error(33);
  const double medium = error(65);
  const double fine = error(129);
  CHECK(coarse / medium > 25.0);
  CHECK(medium / fine > 27.0);
}

TEST_CASE("D10-5 strided dissipation matches contiguous dispatch") {
  constexpr std::size_t points = 29;
  constexpr std::size_t stride = 5;
  const teuk::UniformRadialGrid grid(points, -0.1, 1.2);
  std::vector<double> contiguous(points);
  std::vector<double> strided(points * stride, -73.0);
  for (std::size_t i = 0; i < points; ++i) {
    const double x = grid.coordinate(i);
    contiguous[i] = std::sin(7.0 * x) + 0.2 * std::cos(3.0 * x);
    strided[i * stride] = contiguous[i];
  }
  for (std::size_t i = 0; i < points; ++i) {
    const double direct = teuk::radial_compatible_dissipation_at(
        teuk::RadialDiscretization::D105, contiguous.data(), points, i,
        grid.spacing(), 0.07);
    const double with_stride = teuk::radial_compatible_dissipation_at(
        teuk::RadialDiscretization::D105, strided.data(), points, i,
        grid.spacing(), 0.07, stride);
    CHECK_NEAR(with_stride, direct, 2.0e-13);
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
