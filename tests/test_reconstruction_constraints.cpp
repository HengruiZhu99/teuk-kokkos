#include "test_harness.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/ghp.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_independent_reconstruction_diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/reconstruction_constraints.hpp"
#include "teuk/sbp.hpp"

namespace {

struct ConstraintConvergenceNorm {
  double psi3;
  double psi2;
};

teuk::Complex radial_profile(const double radius, const double wave_number,
                             const teuk::Complex amplitude) {
  return amplitude *
         (1.0 + 0.2 * radius + std::sin(wave_number * radius));
}

teuk::Complex radial_profile_derivative(
    const double radius, const double wave_number,
    const teuk::Complex amplitude) {
  return amplitude * (0.2 + wave_number * std::cos(wave_number * radius));
}

ConstraintConvergenceNorm radial_constraint_norm(
    const std::size_t point_count) {
  const teuk::UniformRadialGrid grid(point_count, 0.1, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.47, 1.6};
  constexpr double cosine = 0.37;
  constexpr double sine = 0.929031931;
  constexpr int m = 1;
  std::vector<teuk::Complex> F(point_count), G(point_count);
  std::vector<teuk::Complex> dr_F(point_count), dr_G(point_count);
  for (std::size_t radial = 0; radial < point_count; ++radial) {
    const double coordinate = grid.coordinate(radial);
    F[radial] = radial_profile(coordinate, 1.31, {0.7, -0.2});
    G[radial] = radial_profile(coordinate, 1.57, {-0.3, 0.5});
  }
  teuk::d42_first_derivative(grid, F, dr_F);
  teuk::d42_first_derivative(grid, G, dr_G);

  double sum3 = 0.0;
  double sum2 = 0.0;
  for (std::size_t radial = 0; radial < point_count; ++radial) {
    const double coordinate = grid.coordinate(radial);
    const auto background = teuk::kerr_background_point(
        parameters, coordinate, cosine, sine);
    const teuk::Complex H =
        radial_profile(coordinate, 1.11, {0.4, 0.1});
    const teuk::Complex dt_F =
        radial_profile(coordinate, 0.83, {-0.12, 0.07});
    const teuk::Complex dt_G =
        radial_profile(coordinate, 0.91, {0.09, -0.11});
    const teuk::Complex ethprime_G =
        radial_profile(coordinate, 1.03, {0.13, 0.17});
    const teuk::Complex ethprime_H =
        radial_profile(coordinate, 1.23, {-0.19, 0.05});
    const teuk::Complex exact_thorn_F = teuk::thorn_n_point(
        F[radial], dt_F,
        radial_profile_derivative(coordinate, 1.31, {0.7, -0.2}), 1, -2,
        -2, m, coordinate, cosine, parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    const teuk::Complex exact_thorn_G = teuk::thorn_n_point(
        G[radial], dt_G,
        radial_profile_derivative(coordinate, 1.57, {-0.3, 0.5}), 2, -1,
        -1, m, coordinate, cosine, parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    const double radius2 = coordinate * coordinate;
    const teuk::Complex Lambda =
        (coordinate * ethprime_G +
         4.0 * radius2 * background.pi0 * G[radial] - exact_thorn_F +
         background.rho0 * F[radial]) /
        (3.0 * radius2 * background.psi20);
    const teuk::Complex Pi =
        (-coordinate * ethprime_H -
         3.0 * radius2 * background.pi0 * H + exact_thorn_G -
         2.0 * background.rho0 * G[radial]) /
        (3.0 * radius2 * background.psi20);
    const teuk::Complex numerical_thorn_F = teuk::thorn_n_point(
        F[radial], dt_F, dr_F[radial], 1, -2, -2, m, coordinate, cosine,
        parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    const teuk::Complex numerical_thorn_G = teuk::thorn_n_point(
        G[radial], dt_G, dr_G[radial], 2, -1, -1, m, coordinate, cosine,
        parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    const auto residual = teuk::independent_reconstruction_constraints_point(
        coordinate, background, F[radial], G[radial], H, Lambda, Pi,
        teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0),
        teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0),
        numerical_thorn_F, numerical_thorn_G, ethprime_G, ethprime_H);
    sum3 += std::norm(std::complex<double>(residual.psi3_bianchi.real(),
                                           residual.psi3_bianchi.imag()));
    sum2 += std::norm(std::complex<double>(residual.psi2_bianchi.real(),
                                           residual.psi2_bianchi.imag()));
  }
  return {std::sqrt(sum3 / static_cast<double>(point_count)),
          std::sqrt(sum2 / static_cast<double>(point_count))};
}

ConstraintConvergenceNorm angular_constraint_norm(const int ell_max) {
  constexpr int m = 1;
  constexpr int ell = 2;
  constexpr double radius = 0.43;
  const int nodes = 2 * ell_max + 5;
  const teuk::KerrParameters parameters{1.0, 0.71, 1.5};
  const teuk::angular::SpinWeightedTransform F_transform(-2, m, ell_max,
                                                         nodes);
  const teuk::angular::SpinWeightedTransform G_transform(-1, m, ell_max,
                                                         nodes);
  std::vector<teuk::Complex> F_modal(F_transform.mode_count(), {0.0, 0.0});
  std::vector<teuk::Complex> G_modal(G_transform.mode_count(), {0.0, 0.0});
  F_modal[static_cast<std::size_t>(ell - F_transform.ell_min())] =
      teuk::Complex(0.7, -0.2);
  G_modal[static_cast<std::size_t>(ell - G_transform.ell_min())] =
      teuk::Complex(-0.3, 0.5);
  const auto F = F_transform.synthesize(F_modal);
  const auto G = G_transform.synthesize(G_modal);
  std::vector<teuk::Complex> H(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> Lambda_exact(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> Pi_exact(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> ethprime_G(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> ethprime_H(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> thorn_F(static_cast<std::size_t>(nodes));
  std::vector<teuk::Complex> thorn_G(static_cast<std::size_t>(nodes));
  for (int node = 0; node < nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double theta = F_transform.grid().theta(index);
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    H[index] = teuk::Complex(0.4, 0.1) *
               teuk::angular::spin_weighted_harmonic_theta(ell, m, 0,
                                                            theta);
    const teuk::Complex dt_F = teuk::Complex(-0.13, 0.07) * F[index];
    const teuk::Complex dt_G = teuk::Complex(0.09, -0.11) * G[index];
    const teuk::Complex lowered_G =
        teuk::Complex(-0.3, 0.5) *
        teuk::angular::lowering_factor(ell, -1) *
        teuk::angular::spin_weighted_harmonic_theta(ell, m, -2, theta);
    const teuk::Complex lowered_H =
        teuk::Complex(0.4, 0.1) *
        teuk::angular::lowering_factor(ell, 0) *
        teuk::angular::spin_weighted_harmonic_theta(ell, m, -1, theta);
    ethprime_G[index] = teuk::ethprime_n_point(
        G[index], dt_G, lowered_G, -1, -1, radius, sine, cosine,
        parameters.spin, parameters.compactification_length);
    ethprime_H[index] = teuk::ethprime_n_point(
        H[index], teuk::Complex(-0.04, 0.06) * H[index], lowered_H, 0, 0,
        radius, sine, cosine, parameters.spin,
        parameters.compactification_length);
    const auto background = teuk::kerr_background_point(
        parameters, radius, cosine, sine);
    thorn_F[index] = teuk::thorn_n_point(
        F[index], dt_F, teuk::Complex(0.17, -0.03) * F[index], 1, -2, -2,
        m, radius, cosine, parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    thorn_G[index] = teuk::thorn_n_point(
        G[index], dt_G, teuk::Complex(-0.08, 0.12) * G[index], 2, -1, -1,
        m, radius, cosine, parameters.mass, parameters.spin,
        parameters.compactification_length, background.epsilon0);
    const double radius2 = radius * radius;
    Lambda_exact[index] =
        (radius * ethprime_G[index] +
         4.0 * radius2 * background.pi0 * G[index] - thorn_F[index] +
         background.rho0 * F[index]) /
        (3.0 * radius2 * background.psi20);
    Pi_exact[index] =
        (-radius * ethprime_H[index] -
         3.0 * radius2 * background.pi0 * H[index] + thorn_G[index] -
         2.0 * background.rho0 * G[index]) /
        (3.0 * radius2 * background.psi20);
  }
  const auto Lambda =
      F_transform.synthesize(F_transform.analyze(Lambda_exact));
  const auto Pi = G_transform.synthesize(G_transform.analyze(Pi_exact));
  double sum3 = 0.0;
  double sum2 = 0.0;
  for (int node = 0; node < nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double theta = F_transform.grid().theta(index);
    const auto background =
        teuk::kerr_background_point(parameters, radius, theta);
    const auto residual = teuk::independent_reconstruction_constraints_point(
        radius, background, F[index], G[index], H[index], Lambda[index],
        Pi[index], teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0),
        teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0), thorn_F[index],
        thorn_G[index], ethprime_G[index], ethprime_H[index]);
    const double magnitude3 = Kokkos::abs(residual.psi3_bianchi);
    const double magnitude2 = Kokkos::abs(residual.psi2_bianchi);
    sum3 += magnitude3 * magnitude3;
    sum2 += magnitude2 * magnitude2;
  }
  return {std::sqrt(sum3 / nodes), std::sqrt(sum2 / nodes)};
}

}  // namespace

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
  pipeline.evaluate_source_activation_on_accepted_state(execution, 0.0);
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
      teuk::SecondOrderSourceMode::ConstraintAware, 0.2, 1.0e6, 1};
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

  CHECK(!pipeline.evaluate_source_activation_on_accepted_state(execution,
                                                               0.1));
  pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                pipeline.storage().rhs(), 0.1);
  const auto before = diagnostics.sample_pipeline(execution, pipeline);
  CHECK(!before.second_order_source_active);
  CHECK(before.independent_reconstruction_constraint_maximum == 0.0);
  CHECK(before.forcing.maximum == 0.0);
  CHECK(before.source_over_r3.maximum > 1.0e-14);

  CHECK(pipeline.evaluate_source_activation_on_accepted_state(execution, 0.3));
  pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                pipeline.storage().rhs(), 0.3);
  const auto after = diagnostics.sample_pipeline(execution, pipeline);
  CHECK(after.second_order_source_active);
  CHECK(after.independent_reconstruction_constraint_maximum > 0.0);
  CHECK(after.forcing.maximum > 1.0e-14);
}

TEST_CASE("independent Bianchi constraints converge under radial refinement") {
  const auto coarse = radial_constraint_norm(17);
  const auto medium = radial_constraint_norm(33);
  const auto fine = radial_constraint_norm(65);
  CHECK(coarse.psi3 / medium.psi3 > 3.5);
  CHECK(medium.psi3 / fine.psi3 > 3.5);
  CHECK(coarse.psi2 / medium.psi2 > 3.5);
  CHECK(medium.psi2 / fine.psi2 > 3.5);
  CHECK(fine.psi3 < 5.0e-7);
  CHECK(fine.psi2 < 5.0e-7);
}

TEST_CASE("independent Bianchi constraints converge with angular bandlimit") {
  const auto ell2 = angular_constraint_norm(2);
  const auto ell4 = angular_constraint_norm(4);
  const auto ell6 = angular_constraint_norm(6);
  CHECK(ell2.psi3 > ell4.psi3);
  CHECK(ell4.psi3 > ell6.psi3);
  CHECK(ell2.psi2 > ell4.psi2);
  CHECK(ell4.psi2 > ell6.psi2);
  CHECK(ell6.psi3 < 6.0e-7);
  CHECK(ell6.psi2 < 8.0e-7);
}
