#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/sbp.hpp"

namespace {

struct HostNormAccumulator {
  double sum_squared = 0.0;
  double max_squared = 0.0;

  void add(const teuk::Complex value) {
    const double norm = value.real() * value.real() +
                        value.imag() * value.imag();
    sum_squared += norm;
    max_squared = std::max(max_squared, norm);
  }

  [[nodiscard]] teuk::PipelineNorm finish(const std::size_t count) const {
    return {std::sqrt(sum_squared / static_cast<double>(count)),
            std::sqrt(max_squared)};
  }
};

template <class HostView>
teuk::Complex host_dense_d42(const HostView& state, const std::size_t mode,
                             const std::size_t field,
                             const std::size_t radial,
                             const std::size_t theta,
                             const teuk::UniformRadialGrid& grid) {
  teuk::Complex derivative(0.0, 0.0);
  for (std::size_t column = 0; column < grid.size(); ++column) {
    derivative += teuk::d42_derivative_coefficient(grid.size(), radial,
                                                    column) *
                  state(mode, field, column, theta);
  }
  return derivative / grid.spacing();
}

void check_norm(const teuk::PipelineNorm& actual,
                const teuk::PipelineNorm& expected,
                const double tolerance = 3.0e-13) {
  CHECK_NEAR(actual.rms, expected.rms, tolerance);
  CHECK_NEAR(actual.maximum, expected.maximum, tolerance);
}

}  // namespace

TEST_CASE("pipeline diagnostics match independent host reductions") {
  constexpr std::size_t mode_count = 3;
  constexpr std::size_t radial_count = 9;
  constexpr std::size_t theta_count = 5;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(radial_count, 0.0, 0.8);
  teuk::SpatialPipelineStorage storage(
      registry, radial_grid, teuk::angular::gauss_legendre(theta_count),
      "diagnostics_host_oracle");
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      source("diagnostics_source", mode_count, radial_count, theta_count);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      forcing("diagnostics_forcing", mode_count, radial_count, theta_count);

  auto host_state = Kokkos::create_mirror_view(storage.state());
  auto host_rhs = Kokkos::create_mirror_view(storage.rhs());
  auto host_source = Kokkos::create_mirror_view(source);
  auto host_forcing = Kokkos::create_mirror_view(forcing);
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          host_state(mode, field, radial, theta) = teuk::Complex(
              0.11 * static_cast<double>(field + 1) + 0.07 * mode +
                  0.013 * radial + 0.003 * theta,
              -0.08 * static_cast<double>(field + 1) + 0.019 * mode -
                  0.005 * radial + 0.002 * theta);
          host_rhs(mode, field, radial, theta) = teuk::Complex(
              -0.031 * static_cast<double>(field + 1) + 0.017 * mode -
                  0.009 * radial + 0.004 * theta,
              0.043 * static_cast<double>(field + 1) - 0.021 * mode +
                  0.006 * radial + 0.001 * theta);
        }
      }
    }
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        host_source(mode, radial, theta) = teuk::Complex(
            0.2 + 0.03 * mode - 0.02 * radial + 0.007 * theta,
            -0.1 + 0.01 * mode + 0.008 * radial - 0.004 * theta);
        host_forcing(mode, radial, theta) = teuk::Complex(
            -0.3 + 0.02 * mode + 0.011 * radial - 0.005 * theta,
            0.15 - 0.006 * mode + 0.003 * radial + 0.009 * theta);
      }
    }
  }
  Kokkos::deep_copy(storage.state(), host_state);
  Kokkos::deep_copy(storage.rhs(), host_rhs);
  Kokkos::deep_copy(source, host_source);
  Kokkos::deep_copy(forcing, host_forcing);

  teuk::PipelineDiagnostics diagnostics(mode_count, radial_grid, theta_count,
                                        "diagnostics_host_oracle");
  const teuk::ExecutionSpace execution;
  const auto report =
      diagnostics.sample_storage(execution, storage, source, forcing);
  const auto repeated =
      diagnostics.sample_storage(execution, storage, source, forcing);

  const std::size_t point_count = mode_count * radial_count * theta_count;
  std::array<HostNormAccumulator, teuk::point_pipeline_field_count>
      state_accumulator{};
  std::array<HostNormAccumulator, teuk::point_pipeline_field_count>
      rhs_accumulator{};
  HostNormAccumulator first_constraint;
  HostNormAccumulator second_constraint;
  HostNormAccumulator source_accumulator;
  HostNormAccumulator forcing_accumulator;
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
             ++field) {
          state_accumulator[field].add(
              host_state(mode, field, radial, theta));
          rhs_accumulator[field].add(host_rhs(mode, field, radial, theta));
        }
        first_constraint.add(
            host_state(mode,
                       static_cast<std::size_t>(teuk::PipelineField::FirstQ),
                       radial, theta) -
            host_dense_d42(
                host_state, mode,
                static_cast<std::size_t>(teuk::PipelineField::FirstPsi),
                radial, theta, radial_grid));
        second_constraint.add(
            host_state(mode,
                       static_cast<std::size_t>(teuk::PipelineField::SecondQ),
                       radial, theta) -
            host_dense_d42(
                host_state, mode,
                static_cast<std::size_t>(teuk::PipelineField::SecondPsi),
                radial, theta, radial_grid));
        source_accumulator.add(host_source(mode, radial, theta));
        forcing_accumulator.add(host_forcing(mode, radial, theta));
      }
    }
  }

  CHECK(report.point_count == point_count);
  CHECK(report.all_finite);
  CHECK(report.scri_finite);
  CHECK(report.horizon_finite);
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    check_norm(report.fields[field].state,
               state_accumulator[field].finish(point_count));
    check_norm(report.fields[field].rhs,
               rhs_accumulator[field].finish(point_count));
    check_norm(repeated.fields[field].state, report.fields[field].state);
    check_norm(repeated.fields[field].rhs, report.fields[field].rhs);
  }
  check_norm(report.first_reduction_constraint,
             first_constraint.finish(point_count));
  check_norm(report.second_reduction_constraint,
             second_constraint.finish(point_count));
  check_norm(report.source_over_r3, source_accumulator.finish(point_count));
  check_norm(report.forcing, forcing_accumulator.finish(point_count));
}

TEST_CASE("pipeline diagnostics reject extent mismatch before launching") {
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(8, 0.0, 0.7);
  teuk::SpatialPipelineStorage storage(
      registry, radial_grid, teuk::angular::gauss_legendre(5),
      "diagnostics_bad_extent");
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      source("diagnostics_good_source", 3, 8, 5);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      bad_forcing("diagnostics_bad_forcing", 3, 8, 4);
  teuk::PipelineDiagnostics diagnostics(3, radial_grid, 5,
                                        "diagnostics_bad_extent");

  bool rejected = false;
  try {
    static_cast<void>(diagnostics.sample_storage(
        teuk::ExecutionSpace{}, storage, source, bad_forcing));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("pipeline diagnostics fail closed on nonfinite endpoint values") {
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(8, 0.0, 0.7);
  teuk::SpatialPipelineStorage storage(
      registry, radial_grid, teuk::angular::gauss_legendre(5),
      "diagnostics_nonfinite");
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      source("diagnostics_finite_source", 3, 8, 5);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, teuk::MemorySpace>
      forcing("diagnostics_finite_forcing", 3, 8, 5);
  Kokkos::deep_copy(storage.state(), teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(storage.rhs(), teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(source, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(forcing, teuk::Complex(0.0, 0.0));

  auto host_state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.state());
  host_state(1, static_cast<std::size_t>(teuk::PipelineField::U), 0, 2) =
      teuk::Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(storage.state(), host_state);

  teuk::PipelineDiagnostics diagnostics(3, radial_grid, 5,
                                        "diagnostics_nonfinite");
  bool rejected = false;
  try {
    static_cast<void>(diagnostics.sample_storage(
        teuk::ExecutionSpace{}, storage, source, forcing));
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("scri-endpoint") !=
               std::string::npos;
  }
  CHECK(rejected);
}

