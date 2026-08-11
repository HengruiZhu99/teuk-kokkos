#include "test_harness.hpp"

#include <cmath>
#include <cstddef>

#include "teuk/ghp.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_independent_reconstruction_diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/reconstruction_constraints.hpp"

TEST_CASE("thorn point operator matches the independent legacy expansion") {
  const teuk::KerrParameters parameters{1.0, 0.63, 1.7};
  constexpr double radius = 0.37;
  constexpr double cosine = 0.41;
  const auto background = teuk::kerr_background_point(
      parameters, radius, cosine, std::sqrt(1.0 - cosine * cosine));
  const teuk::Complex value(0.7, -0.2);
  const teuk::Complex dt(-0.4, 0.9);
  const teuk::Complex dr(0.3, 0.5);
  const int n = 2;
  const int spin = -1;
  const int boost = -1;
  const int m = 2;
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double denominator =
      length2 * length2 + parameters.spin * parameters.spin * radius *
                              radius * cosine * cosine;
  const teuk::Complex expected =
      radius * 2.0 * parameters.mass *
          (2.0 * parameters.mass -
           parameters.spin * parameters.spin * radius / length2) /
          denominator *
          dt -
      0.5 *
          (length2 - 2.0 * parameters.mass * radius +
           parameters.spin * parameters.spin * radius * radius / length2) /
          denominator *
          (radius * dr + static_cast<double>(n) * value) +
      radius * teuk::Complex(0.0, 1.0) * parameters.spin *
          static_cast<double>(m) / denominator * value -
      radius *
          (static_cast<double>(spin + boost) * background.epsilon0 +
           static_cast<double>(-spin + boost) *
               Kokkos::conj(background.epsilon0)) *
          value;
  CHECK_COMPLEX_NEAR(
      teuk::thorn_n_point(value, dt, dr, n, spin, boost, m, radius, cosine,
                          parameters.mass, parameters.spin,
                          parameters.compactification_length,
                          background.epsilon0),
      expected, 3.0e-15);
}

TEST_CASE("independent Bianchi and hll residuals match legacy equations") {
  const teuk::KerrParameters parameters{1.0, 0.52, 1.4};
  constexpr double radius = 0.43;
  constexpr double theta = 0.81;
  const auto background =
      teuk::kerr_background_point(parameters, radius, theta);
  const teuk::Complex F(0.2, -0.7), G(-0.1, 0.4), H(0.5, 0.3);
  const teuk::Complex Lambda(-0.6, 0.2), Pi(0.8, -0.1), B(0.3, 0.9);
  const teuk::Complex C(-0.2, -0.5), U(0.7, 0.1), U_sharp(-0.4, 0.6);
  const teuk::Complex thorn_F(0.11, -0.13), thorn_G(-0.17, 0.19);
  const teuk::Complex ethprime_G(0.23, 0.29), ethprime_H(-0.31, 0.37);
  const auto actual = teuk::independent_reconstruction_constraints_point(
      radius, background, F, G, H, Lambda, Pi, B, C, U, U_sharp,
      thorn_F, thorn_G, ethprime_G, ethprime_H);
  const double radius2 = radius * radius;
  const double radius3 = radius2 * radius;
  const teuk::Complex expected3 =
      radius * ethprime_G + 4.0 * radius2 * background.pi0 * G - thorn_F +
      background.rho0 * F - 3.0 * radius2 * background.psi20 * Lambda;
  const teuk::Complex expected2 =
      background.psi20 *
          (-3.0 * radius3 * background.mu0 * C -
           1.5 * radius3 * background.tau0 * B - 3.0 * radius2 * Pi) -
      radius * ethprime_H - 3.0 * radius2 * background.pi0 * H + thorn_G -
      2.0 * background.rho0 * G;
  CHECK_COMPLEX_NEAR(actual.psi3_bianchi, expected3, 3.0e-15);
  CHECK_COMPLEX_NEAR(actual.psi2_bianchi, expected2, 3.0e-15);
  CHECK_COMPLEX_NEAR(actual.hll_reality,
                     U / background.mu0 -
                         U_sharp / Kokkos::conj(background.mu0),
                     3.0e-15);
}

TEST_CASE("default Gaussian exposes inconsistent reconstruction constraints") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 0, 2});
  const teuk::UniformRadialGrid grid(17, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.4, 1.5};
  teuk::SpatialPipeline pipeline(execution, registry, grid, 4, 7, parameters);
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.12;
  pulse.modes = {{2, 2, teuk::Complex(1.0e-3, 0.0)}};
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, 4, parameters, pulse);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  teuk::PipelineIndependentReconstructionDiagnostics diagnostics(
      registry.size(), grid.size(), 7);
  const auto report = diagnostics.sample(execution, pipeline);
  CHECK(report.psi3_bianchi.maximum > 1.0e-8);
  CHECK(report.psi2_bianchi.maximum > 1.0e-8);
  CHECK(report.hll_reality.maximum < 1.0e-13);
  const auto active = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.source_active());
  const auto constraint_max = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.source_constraint_max_squared());
  CHECK(active(0) == 0);
  CHECK(std::sqrt(constraint_max(0)) >= report.psi3_bianchi.maximum);
}

TEST_CASE("causal constraint-aware source gate is visible and configurable") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.35, 1.5};
  const teuk::SecondOrderSourcePolicy policy{
      teuk::SecondOrderSourceMode::ConstraintAware, 0.2, 1.0e6};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, 3, 6, parameters, 0.1, 0.0,
      teuk::ReductionEvolution::FreeDamped, "causal_source_gate", policy);
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.15;
  pulse.modes = {{3, -1, teuk::Complex(2.0e-4, -1.0e-4)},
                 {3, 1, teuk::Complex(-1.0e-4, 1.5e-4)}};
  for (auto& scale : pulse.reconstruction_scales) {
    scale = teuk::Complex(0.2, -0.05);
  }
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, 3, parameters, pulse);
  teuk::PipelineDiagnostics diagnostics(registry.size(), grid, 6);

  pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                pipeline.storage().rhs(), 0.1);
  const auto before = diagnostics.sample_pipeline(execution, pipeline);
  CHECK(!before.second_order_source_active);
  CHECK(before.independent_reconstruction_constraint_maximum > 0.0);
  CHECK(before.forcing.maximum == 0.0);
  CHECK(before.source_over_r3.maximum > 1.0e-14);

  pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                pipeline.storage().rhs(), 0.3);
  const auto after = diagnostics.sample_pipeline(execution, pipeline);
  CHECK(after.second_order_source_active);
  CHECK(after.independent_reconstruction_constraint_maximum ==
        before.independent_reconstruction_constraint_maximum);
  CHECK(after.forcing.maximum > 1.0e-14);
}
