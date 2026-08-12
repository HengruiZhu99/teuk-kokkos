#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "teuk/pipeline_initial_data.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace {

constexpr int activation_ell_max = 3;
constexpr int activation_theta_nodes = 6;

teuk::PipelineGaussianPulse activation_pulse(const double scale = 1.0) {
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.16;
  pulse.modes = {
      {3, -1, scale * teuk::Complex(1.7e-4, -0.8e-4)},
      {3, 1, scale * teuk::Complex(-1.1e-4, 1.3e-4)}};
  for (std::size_t field = 0; field < pulse.reconstruction_scales.size();
       ++field) {
    pulse.reconstruction_scales[field] =
        teuk::Complex(0.11 + 0.013 * static_cast<double>(field),
                      -0.04 + 0.007 * static_cast<double>(field));
  }
  return pulse;
}

std::vector<teuk::Complex> snapshot(
    const teuk::ExecutionSpace& execution,
    const teuk::SpatialPipeline& pipeline) {
  Kokkos::View<teuk::Complex*, Kokkos::HostSpace> host(
      "source_activation_snapshot", pipeline.storage().value_count());
  Kokkos::deep_copy(execution, host, pipeline.storage().flat_state());
  execution.fence("copy source activation test state");
  std::vector<teuk::Complex> values(host.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  return values;
}

std::vector<teuk::Complex> event_trajectory(const int steps) {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.31, 1.5};
  constexpr double source_start = 0.0027;
  constexpr double final_time = 0.008;
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, activation_ell_max, activation_theta_nodes,
      background, 0.1, 0.0, teuk::ReductionEvolution::FreeDamped,
      "source_event_convergence_" + std::to_string(steps),
      {teuk::SecondOrderSourceMode::Unrestricted, source_start, 0.0, 1});
  auto pulse = activation_pulse();
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, activation_ell_max, background, pulse);
  const double dt = final_time / static_cast<double>(steps);
  double time = 0.0;
  for (int step = 0; step < steps; ++step) {
    pipeline.step(execution, time, dt);
    time += dt;
  }
  execution.fence("finish source event convergence trajectory");
  CHECK(pipeline.source_activation_state().active);
  CHECK_NEAR(pipeline.source_activation_state().activation_time, source_start,
             0.0);
  return snapshot(execution, pipeline);
}

double second_order_relative_difference(
    const std::vector<teuk::Complex>& left,
    const std::vector<teuk::Complex>& right) {
  CHECK(left.size() == right.size());
  constexpr std::size_t mode_count = 3;
  constexpr std::size_t field_count = teuk::point_pipeline_field_count;
  constexpr std::size_t radial_count = 9;
  constexpr std::size_t theta_count = activation_theta_nodes;
  double difference_squared = 0.0;
  double reference_squared = 0.0;
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    for (std::size_t field =
             static_cast<std::size_t>(teuk::PipelineField::SecondP);
         field <= static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
         ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          const std::size_t flat =
              ((mode * field_count + field) * radial_count + radial) *
                  theta_count +
              theta;
          const double difference = Kokkos::abs(left[flat] - right[flat]);
          const double reference = Kokkos::abs(right[flat]);
          difference_squared += difference * difference;
          reference_squared += reference * reference;
        }
      }
    }
  }
  return std::sqrt(difference_squared / reference_squared);
}

}  // namespace

TEST_CASE("latched source state is fixed across every RK substep stage") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.31, 1.5};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, activation_ell_max, activation_theta_nodes,
      background, 0.1, 0.0, teuk::ReductionEvolution::FreeDamped,
      "source_stage_consistency",
      {teuk::SecondOrderSourceMode::Unrestricted, 0.03, 0.0, 1});
  auto pulse = activation_pulse();
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, activation_ell_max, background, pulse);

  pipeline.step(execution, 0.0, 0.05);
  const auto stages = pipeline.last_step_stage_source_states();
  CHECK(stages.size() == 8);
  for (std::size_t stage = 0; stage < 4; ++stage) CHECK(stages[stage] == 0);
  for (std::size_t stage = 4; stage < 8; ++stage) CHECK(stages[stage] == 1);
  CHECK(pipeline.source_activation_state().active);
  CHECK_NEAR(pipeline.source_activation_state().activation_time, 0.03, 0.0);
}

TEST_CASE("source event at an endpoint contributes nothing to the preceding step") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.31, 1.5};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, activation_ell_max, activation_theta_nodes,
      background, 0.1, 0.0, teuk::ReductionEvolution::FreeDamped,
      "source_endpoint_activation",
      {teuk::SecondOrderSourceMode::Unrestricted, 0.01, 0.0, 1});
  auto pulse = activation_pulse();
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, activation_ell_max, background, pulse);

  pipeline.step(execution, 0.0, 0.01);
  const auto stages = pipeline.last_step_stage_source_states();
  CHECK(stages.size() == 4);
  for (const int active : stages) CHECK(active == 0);
  CHECK(pipeline.source_activation_state().active);
  CHECK_NEAR(pipeline.source_activation_state().activation_time, 0.01, 0.0);

  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  for (std::size_t mode = 0; mode < host.extent(0); ++mode) {
    for (std::size_t field =
             static_cast<std::size_t>(teuk::PipelineField::SecondP);
         field <= static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
         ++field) {
      for (std::size_t radial = 0; radial < host.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < host.extent(3); ++theta) {
          CHECK_COMPLEX_NEAR(host(mode, field, radial, theta),
                             teuk::Complex(0.0, 0.0), 0.0);
        }
      }
    }
  }
}

TEST_CASE("exact source event splitting retains fourth-order convergence") {
  const auto coarse = event_trajectory(2);
  const auto medium = event_trajectory(4);
  const auto fine = event_trajectory(8);
  const auto reference = event_trajectory(64);
  const double coarse_error =
      second_order_relative_difference(coarse, reference);
  const double medium_error =
      second_order_relative_difference(medium, reference);
  const double fine_error = second_order_relative_difference(fine, reference);
  CHECK(coarse_error / medium_error > 10.0);
  CHECK(medium_error / fine_error > 10.0);
}

TEST_CASE("normalized source eligibility is invariant under amplitude scaling") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.31, 1.5};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, activation_ell_max, activation_theta_nodes,
      background, 0.1, 0.0, teuk::ReductionEvolution::FreeDamped,
      "source_amplitude_invariance",
      {teuk::SecondOrderSourceMode::ConstraintAware, 0.0, 0.0, 1});

  std::array<teuk::SourceConstraintNorms, 3> reports;
  constexpr std::array<double, 3> scales{1.0e-5, 1.0, 1.0e5};
  for (std::size_t i = 0; i < scales.size(); ++i) {
    auto pulse = activation_pulse(scales[i]);
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, activation_ell_max, background, pulse);
    CHECK(!pipeline.evaluate_source_activation_on_accepted_state(execution,
                                                                 0.0));
    reports[i] = pipeline.source_constraint_norms();
    CHECK(reports[i].all_finite);
    CHECK(reports[i].controlling_normalized > 0.0);
  }
  CHECK_NEAR(reports[0].controlling_normalized,
             reports[1].controlling_normalized, 3.0e-12);
  CHECK_NEAR(reports[2].controlling_normalized,
             reports[1].controlling_normalized, 3.0e-12);
  CHECK(reports[0].psi3_bianchi.absolute_maximum <
        reports[1].psi3_bianchi.absolute_maximum);
  CHECK(reports[1].psi3_bianchi.absolute_maximum <
        reports[2].psi3_bianchi.absolute_maximum);
}

TEST_CASE("D8-4 source constraint gate uses the selected SBP norm") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(17, 0.0, 0.8);
  const auto angular_grid = teuk::angular::gauss_legendre(5);
  teuk::SourceConstraintEvaluator evaluator(
      1, grid, angular_grid, "d84_source_constraint_norm",
      teuk::RadialDiscretization::D84);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      residuals("d84_source_constraint_residuals", 1, 3, grid.size(), 5);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      terms("d84_source_constraint_terms", 1, 6, grid.size(), 5);
  Kokkos::deep_copy(residuals, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(terms, teuk::Complex(1.0, 0.0));
  auto host_residuals = Kokkos::create_mirror_view(residuals);
  Kokkos::deep_copy(host_residuals, teuk::Complex(0.0, 0.0));
  for (std::size_t theta = 0; theta < 5; ++theta) {
    host_residuals(0, 0, 0, theta) = teuk::Complex(1.0, 0.0);
  }
  Kokkos::deep_copy(execution, residuals, host_residuals);
  const auto report = evaluator.sample(execution, residuals, terms);
  const double expected = std::sqrt(
      teuk::radial_norm_weight(teuk::RadialDiscretization::D84,
                               grid.size(), 0) /
      static_cast<double>(grid.size() - 1));
  CHECK_NEAR(report.psi3_bianchi.weighted_rms, expected, 2.0e-14);
  CHECK(report.all_finite);

  bool rejected = false;
  try {
    const teuk::UniformRadialGrid too_small(15, 0.0, 0.8);
    teuk::SourceConstraintEvaluator invalid(
        1, too_small, angular_grid, "undersized_d84_source_constraint_norm",
        teuk::RadialDiscretization::D84);
    static_cast<void>(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}
