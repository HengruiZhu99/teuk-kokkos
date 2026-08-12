#include "test_harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "teuk/radial_discretization.hpp"
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
  teuk::d105_first_derivative(grid, values, derivative);
  double error = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    error = std::max(
        error,
        std::abs(derivative[i] - smooth_derivative(grid.coordinate(i))));
  }
  return error;
}

struct DissipativeManufacturedError {
  double rhs_maximum = 0.0;
  double dissipation_endpoint = 0.0;
  double differentiated_dissipation_endpoint = 0.0;
};

DissipativeManufacturedError dissipative_manufactured_error(
    const std::size_t points) {
  constexpr double strength = 0.006;
  const teuk::UniformRadialGrid grid(points, 0.0, 1.0);
  std::vector<double> values(points);
  std::vector<double> derivative(points);
  std::vector<double> dissipation(points);
  std::vector<double> differentiated_dissipation(points);
  for (std::size_t i = 0; i < points; ++i) {
    values[i] = std::pow(grid.coordinate(i), 6);
  }
  teuk::d105_first_derivative(grid, values, derivative);
  DissipativeManufacturedError result;
  for (std::size_t i = 0; i < points; ++i) {
    dissipation[i] = teuk::d105_compatible_dissipation_at(
        values.data(), points, i, grid.spacing(), strength);
    const double exact = 6.0 * std::pow(grid.coordinate(i), 5);
    result.rhs_maximum =
        std::max(result.rhs_maximum,
                 std::abs(derivative[i] + dissipation[i] - exact));
    if (i == 0) result.dissipation_endpoint = std::abs(dissipation[i]);
  }
  teuk::d105_first_derivative(grid, dissipation, differentiated_dissipation);
  result.differentiated_dissipation_endpoint =
      std::abs(differentiated_dissipation.front());
  return result;
}

}  // namespace

TEST_CASE("D10-5 diagonal norm and derivative satisfy the SBP identity") {
  for (const std::size_t points : {22U, 23U, 31U, 48U}) {
    const teuk::UniformRadialGrid grid(points, 0.0, 1.4);
    double residual = 0.0;
    for (std::size_t i = 0; i < points; ++i) {
      CHECK(teuk::d105_norm_matrix_entry(grid, i, i) > 0.0);
      for (std::size_t j = 0; j < points; ++j) {
        const double hd =
            teuk::d105_norm_matrix_entry(grid, i, i) *
                teuk::d105_derivative_matrix_entry(grid, i, j) +
            teuk::d105_derivative_matrix_entry(grid, j, i) *
                teuk::d105_norm_matrix_entry(grid, j, j);
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

TEST_CASE("D10-5 is quintic-exact globally and degree-ten exact inside") {
  const teuk::UniformRadialGrid grid(49, -0.25, 1.1);
  std::vector<double> values(grid.size());
  std::vector<double> derivative(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    values[i] = 0.3 - 0.7 * x + 0.4 * x * x - 0.2 * x * x * x +
                0.1 * std::pow(x, 4) - 0.03 * std::pow(x, 5);
  }
  teuk::d105_first_derivative(grid, values, derivative);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(derivative[i],
               -0.7 + 0.8 * x - 0.6 * x * x + 0.4 * std::pow(x, 3) -
                   0.15 * std::pow(x, 4),
               2.0e-10);
  }

  for (std::size_t i = 0; i < grid.size(); ++i) {
    values[i] = std::pow(grid.coordinate(i), 10);
  }
  teuk::d105_first_derivative(grid, values, derivative);
  for (std::size_t i = teuk::d105_boundary_width;
       i + teuk::d105_boundary_width < grid.size(); ++i) {
    CHECK_NEAR(derivative[i], 10.0 * std::pow(grid.coordinate(i), 9),
               2.0e-9);
  }
}

TEST_CASE("D10-5 has fifth-order maximum-norm boundary convergence") {
  const double coarse = maximum_error(33);
  const double medium = maximum_error(65);
  const double fine = maximum_error(129);
  CHECK(coarse / medium > 25.0);
  CHECK(medium / fine > 27.0);
}

TEST_CASE("D10-5 compatible dissipation is SBP-negative and quintic exact") {
  const teuk::UniformRadialGrid grid(31, -0.2, 1.0);
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
    dissipation[i] = teuk::d105_compatible_dissipation_at(
        values.data(), grid.size(), i, grid.spacing(), strength);
  }
  teuk::Complex energy = 0.0;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    energy += grid.spacing() * teuk::d105_norm_weight(grid.size(), i) *
              Kokkos::conj(values[i]) * dissipation[i];
  }
  double expected = 0.0;
  for (std::size_t i = 0; i + 6 < grid.size(); ++i) {
    const teuk::Complex difference =
        values[i] - 6.0 * values[i + 1] + 15.0 * values[i + 2] -
        20.0 * values[i + 3] + 15.0 * values[i + 4] -
        6.0 * values[i + 5] + values[i + 6];
    expected -= strength * Kokkos::abs(difference) * Kokkos::abs(difference);
  }
  CHECK(energy.real() <= 0.0);
  CHECK_NEAR(energy.real(), expected, 4.0e-12);
  CHECK_NEAR(energy.imag(), 0.0, 4.0e-12);

  std::vector<double> quintic(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    quintic[i] = 0.2 - x + 0.3 * x * x - 0.4 * std::pow(x, 3) +
                 0.7 * std::pow(x, 4) - 0.2 * std::pow(x, 5);
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK(std::abs(teuk::d105_compatible_dissipation_at(
              quintic.data(), grid.size(), i, grid.spacing(), strength)) <
          5.0e-8);
  }
}

TEST_CASE("D10-5 nonzero-dissipation manufactured RHS retains closure order") {
  const auto coarse = dissipative_manufactured_error(25);
  const auto medium = dissipative_manufactured_error(49);
  const auto fine = dissipative_manufactured_error(97);
  const double rhs_coarse_medium = coarse.rhs_maximum / medium.rhs_maximum;
  const double rhs_medium_fine = medium.rhs_maximum / fine.rhs_maximum;
  const double dissipation_coarse_medium =
      coarse.dissipation_endpoint / medium.dissipation_endpoint;
  const double dissipation_medium_fine =
      medium.dissipation_endpoint / fine.dissipation_endpoint;
  const double differentiated_coarse_medium =
      coarse.differentiated_dissipation_endpoint /
      medium.differentiated_dissipation_endpoint;
  const double differentiated_medium_fine =
      medium.differentiated_dissipation_endpoint /
      fine.differentiated_dissipation_endpoint;
  std::cout << "D105 manufactured epsilon=0.006 RHS errors "
            << coarse.rhs_maximum << " " << medium.rhs_maximum << " "
            << fine.rhs_maximum << " ratios " << rhs_coarse_medium << " "
            << rhs_medium_fine << " endpoint dissipation "
            << coarse.dissipation_endpoint << " "
            << medium.dissipation_endpoint << " "
            << fine.dissipation_endpoint << " ratios "
            << dissipation_coarse_medium << " "
            << dissipation_medium_fine << " differentiated endpoint "
            << coarse.differentiated_dissipation_endpoint << " "
            << medium.differentiated_dissipation_endpoint << " "
            << fine.differentiated_dissipation_endpoint << " ratios "
            << differentiated_coarse_medium << " "
            << differentiated_medium_fine << '\n';
  CHECK(coarse.dissipation_endpoint > 0.0);
  CHECK(std::isfinite(rhs_coarse_medium));
  CHECK(std::isfinite(rhs_medium_fine));
  CHECK(std::isfinite(dissipation_coarse_medium));
  CHECK(std::isfinite(dissipation_medium_fine));
  CHECK(std::isfinite(differentiated_coarse_medium));
  CHECK(std::isfinite(differentiated_medium_fine));
  CHECK(rhs_coarse_medium > 15.0);
  CHECK(rhs_medium_fine > 15.0);
  CHECK(dissipation_coarse_medium > 15.0);
  CHECK(dissipation_medium_fine > 15.0);
  CHECK(differentiated_coarse_medium > 15.0);
  CHECK(differentiated_medium_fine > 15.0);
}

TEST_CASE("D10-5 dissipation exposes its explicit RK4 stability scale") {
  const auto maximum_symmetric_row_sum = [](const std::size_t points) {
    constexpr std::array<double, 7> stencil{1.0, -6.0, 15.0, -20.0,
                                             15.0, -6.0, 1.0};
    const std::size_t rows = points - stencil.size() + 1;
    double maximum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
      double row_sum = 0.0;
      for (std::size_t other = 0; other < rows; ++other) {
        double entry = 0.0;
        for (std::size_t position = 0; position < stencil.size(); ++position) {
          const std::size_t column = row + position;
          if (column >= other && column < other + stencil.size()) {
            entry += stencil[position] * stencil[column - other] /
                     teuk::d105_norm_weight(points, column);
          }
        }
        row_sum += std::abs(entry);
      }
      maximum = std::max(maximum, row_sum);
    }
    return maximum;
  };
  for (std::size_t points = 22; points <= 64; ++points) {
    const double row_sum = maximum_symmetric_row_sum(points);
    CHECK(row_sum > 5497.0);
    CHECK(row_sum < teuk::d105_dissipation_spectral_radius_bound);
  }

  constexpr std::size_t points = 25;
  std::vector<double> iterate(points);
  std::vector<double> applied(points);
  for (std::size_t i = 0; i < points; ++i) {
    iterate[i] = 0.3 + std::sin(1.1 * static_cast<double>(i));
  }
  double largest_eigenvalue = 0.0;
  for (int iteration = 0; iteration < 400; ++iteration) {
    for (std::size_t i = 0; i < points; ++i) {
      applied[i] = -teuk::d105_compatible_dissipation_at(
          iterate.data(), points, i, 1.0, 1.0);
    }
    double norm_squared = 0.0;
    for (std::size_t i = 0; i < points; ++i) {
      norm_squared += teuk::d105_norm_weight(points, i) * applied[i] *
                      applied[i];
    }
    const double inverse_norm = 1.0 / std::sqrt(norm_squared);
    for (std::size_t i = 0; i < points; ++i) {
      iterate[i] = applied[i] * inverse_norm;
    }
  }
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    applied[i] = -teuk::d105_compatible_dissipation_at(
        iterate.data(), points, i, 1.0, 1.0);
    const double weight = teuk::d105_norm_weight(points, i);
    numerator += weight * iterate[i] * applied[i];
    denominator += weight * iterate[i] * iterate[i];
  }
  largest_eigenvalue = numerator / denominator;
  constexpr double rk4_negative_real_extent = 2.785293563405282;
  constexpr double strength = 0.006;
  constexpr double spacing = 0.64 / static_cast<double>(points - 1);
  const double reference_maximum_step =
      rk4_negative_real_extent * spacing / (strength * largest_eigenvalue);
  const double guarded_maximum_step =
      spacing * teuk::radial_dissipation_rk4_maximum_step_ratio(
                    teuk::RadialDiscretization::D105, strength);
  std::cout << "D105 n=25 dimensionless dissipation spectral radius "
            << largest_eigenvalue << " RK4 reference/guarded dt_max"
            << "(epsilon=0.006,h=" << spacing << ") "
            << reference_maximum_step << " " << guarded_maximum_step << '\n';
  CHECK(std::isfinite(largest_eigenvalue));
  CHECK(largest_eigenvalue > 5020.0);
  CHECK(largest_eigenvalue < 5030.0);
  CHECK(reference_maximum_step > 0.00245);
  CHECK(reference_maximum_step < 0.00248);
  CHECK(guarded_maximum_step < reference_maximum_step);
  CHECK(guarded_maximum_step > 0.00179);
  CHECK(guarded_maximum_step < 0.00181);
}

TEST_CASE("D10-5 point derivative and dissipation execute on device") {
  const teuk::UniformRadialGrid grid(24, 0.0, 1.0);
  Kokkos::View<double*> values("d105_values", grid.size());
  Kokkos::View<double*> derivative("d105_derivative", grid.size());
  Kokkos::View<double*> dissipation("d105_dissipation", grid.size());
  auto host = Kokkos::create_mirror_view(values);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    host(i) = 0.5 - 0.3 * x + 0.2 * std::pow(x, 5);
  }
  Kokkos::deep_copy(values, host);
  Kokkos::parallel_for(
      "d105_device_point", grid.size(), KOKKOS_LAMBDA(const int i) {
        const std::size_t index = static_cast<std::size_t>(i);
        derivative(index) = teuk::d105_first_derivative_at(
            values.data(), grid.size(), index, 1.0 / grid.spacing());
        dissipation(index) = teuk::d105_compatible_dissipation_at(
            values.data(), grid.size(), index, grid.spacing(), 0.1);
      });
  const auto host_derivative = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), derivative);
  const auto host_dissipation = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), dissipation);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double x = grid.coordinate(i);
    CHECK_NEAR(host_derivative(i), -0.3 + std::pow(x, 4), 4.0e-10);
    CHECK(std::abs(host_dissipation(i)) < 5.0e-8);
  }
}
