#include "test_harness.hpp"

#include <array>
#include <cstddef>

#include "teuk/horizon_diagnostics.hpp"
#include "teuk/pipeline_storage.hpp"

namespace {

using StateView = Kokkos::View<teuk::Complex****, Kokkos::LayoutRight,
                               teuk::MemorySpace>;

}  // namespace

TEST_CASE("horizon diagnostics recover four transverse polynomial derivatives") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr std::size_t modes = 2;
  constexpr std::size_t theta_count = 3;
  StateView state("horizon_polynomial_state", modes,
                  teuk::point_pipeline_field_count, grid.size(), theta_count);
  auto host = Kokkos::create_mirror_view(state);
  const std::array<double, 5> real_coefficients{0.7, -0.2, 0.05, 0.013,
                                                -0.004};
  const std::array<double, 5> imaginary_coefficients{-0.3, 0.11, -0.027,
                                                     0.006, 0.002};
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const double x = grid.coordinate(radial) - grid.upper_radius();
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        teuk::Complex value(0.0, 0.0);
        double power = 1.0;
        for (std::size_t degree = 0; degree < real_coefficients.size();
             ++degree) {
          const double scale = 1.0 + 0.2 * static_cast<double>(mode) +
                               0.1 * static_cast<double>(theta);
          value += scale * power *
                   teuk::Complex(real_coefficients[degree],
                                 imaginary_coefficients[degree]);
          power *= x;
        }
        host(mode,
             static_cast<std::size_t>(teuk::PipelineField::FirstPsi), radial,
             theta) = value;
      }
    }
  }
  Kokkos::deep_copy(execution, state, host);
  teuk::HorizonTransverseDiagnostics diagnostic(grid, modes, theta_count);
  diagnostic.evaluate(
      execution, state,
      static_cast<std::size_t>(teuk::PipelineField::FirstPsi));
  execution.fence("horizon polynomial derivative test");
  const auto result = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, diagnostic.values());
  const std::array<double, 5> factorial{1.0, 1.0, 2.0, 6.0, 24.0};
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const double scale = 1.0 + 0.2 * static_cast<double>(mode) +
                           0.1 * static_cast<double>(theta);
      for (std::size_t derivative = 0;
           derivative < teuk::horizon_derivative_count; ++derivative) {
        const teuk::Complex expected =
            scale * factorial[derivative] *
            teuk::Complex(real_coefficients[derivative],
                          imaginary_coefficients[derivative]);
        CHECK_COMPLEX_NEAR(result(mode, derivative, theta), expected,
                           2.0e-8);
      }
    }
  }
}

TEST_CASE("horizon diagnostics reject mismatched full-grid state") {
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  teuk::HorizonTransverseDiagnostics diagnostic(grid, 1, 2);
  StateView wrong("wrong_horizon_state", 1,
                  teuk::point_pipeline_field_count, 9, 3);
  bool threw = false;
  try {
    diagnostic.evaluate(teuk::ExecutionSpace{}, wrong, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}
