#include "test_harness.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <complex>
#include <cstdint>

#include "plus2_linear_psi0_oracle_fixtures.hpp"
#include "teuk/boundary.hpp"
#include "teuk/jet.hpp"
#include "teuk/plus2_linear_psi0.hpp"

namespace {

using KC = teuk::Complex;

int linear_psi0_allocations = 0;

void count_linear_psi0_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                  const void*, std::uint64_t) {
  ++linear_psi0_allocations;
}

teuk::Plus2OrgMetricFields scaled(const teuk::Plus2OrgMetricFields& value,
                                  const double factor) {
  return {factor * value.h_ll, factor * value.h_lm, factor * value.h_mm};
}

teuk::Plus2OrgMetricFields shifted(
    const teuk::Plus2OrgMetricFields& value,
    const teuk::Plus2OrgMetricFields& tangent, const double epsilon) {
  return {value.h_ll + epsilon * tangent.h_ll,
          value.h_lm + epsilon * tangent.h_lm,
          value.h_mm + epsilon * tangent.h_mm};
}

teuk::Plus2OrgMetricStage scaled(const teuk::Plus2OrgMetricStage& value,
                                 const double factor) {
  return {scaled(value.h, factor), scaled(value.h_T, factor),
          scaled(value.h_TT, factor)};
}

teuk::Plus2OrgMetricDerivativeSlots scaled(
    const teuk::Plus2OrgMetricDerivativeSlots& value, const double factor) {
  return {scaled(value.h_R, factor),
          scaled(value.h_TR, factor),
          scaled(value.h_RR, factor),
          scaled(value.h_theta, factor),
          scaled(value.h_Ttheta, factor),
          scaled(value.h_Rtheta, factor),
          scaled(value.h_thetatheta, factor),
          scaled(value.h_phi, factor),
          scaled(value.h_Tphi, factor),
          scaled(value.h_Rphi, factor),
          scaled(value.h_thetaphi, factor),
          scaled(value.h_phiphi, factor)};
}

}  // namespace

TEST_CASE("linear Psi0 background reconciles all Ripley Eq8 coefficients") {
  const teuk::KerrParameters parameters{1.0, 0.73, 1.9};
  constexpr double radius = 0.61;
  constexpr double theta = 0.88;
  const double sine = std::sin(theta);
  const double cosine = std::cos(theta);
  const auto actual = teuk::plus2_linear_background_point(
      parameters, radius, sine, cosine);
  const auto existing =
      teuk::kerr_background_point(parameters, radius, cosine, sine);
  CHECK_COMPLEX_NEAR(actual.rho, radius * existing.rho0, 2.0e-15);
  CHECK_COMPLEX_NEAR(actual.epsilon, radius * radius * existing.epsilon0,
                     2.0e-15);
  CHECK_COMPLEX_NEAR(actual.tau, radius * radius * existing.tau0, 2.0e-15);
  CHECK_COMPLEX_NEAR(actual.pi, radius * radius * existing.pi0, 2.0e-15);

  const std::complex<double> i(0.0, 1.0);
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const std::complex<double> dm(length2,
                                -parameters.spin * radius * cosine);
  const std::complex<double> dp(length2,
                                parameters.spin * radius * cosine);
  const std::complex<double> expected_alpha =
      radius * cosine / (2.0 * std::sqrt(2.0) * sine * dp);
  const std::complex<double> expected_beta =
      radius *
      (-length2 * cosine / sine +
       i * parameters.spin * radius * sine * (1.0 / (sine * sine) + 1.0)) /
      (2.0 * std::sqrt(2.0) * dm * dm);
  CHECK_COMPLEX_NEAR(actual.alpha,
                     KC(expected_alpha.real(), expected_alpha.imag()),
                     2.0e-15);
  CHECK_COMPLEX_NEAR(actual.beta,
                     KC(expected_beta.real(), expected_beta.imag()),
                     2.0e-15);
}

TEST_CASE("linear Psi0 matches independent Schwarzschild and Kerr curvature") {
  for (const auto& fixture :
       teuk::test::plus2_linear_psi0::coordinate_oracle_fixtures()) {
    const double sine = std::sin(fixture.theta);
    const double cosine = std::cos(fixture.theta);
    const auto actual = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine, fixture.stage,
        fixture.derivatives);
    CHECK_COMPLEX_NEAR(actual.psi0_code_tetrad, fixture.expected_psi0,
                       8.0e-13);
  }
}

TEST_CASE("linear Psi0 is exactly linear in every metric derivative slot") {
  constexpr double factor = -2.75;
  for (const auto& fixture :
       teuk::test::plus2_linear_psi0::coordinate_oracle_fixtures()) {
    const double sine = std::sin(fixture.theta);
    const double cosine = std::cos(fixture.theta);
    const auto baseline = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine, fixture.stage,
        fixture.derivatives);
    const auto result = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine,
        scaled(fixture.stage, factor), scaled(fixture.derivatives, factor));
    CHECK_COMPLEX_NEAR(result.sigma1, factor * baseline.sigma1, 3.0e-14);
    CHECK_COMPLEX_NEAR(result.kappa1, factor * baseline.kappa1, 3.0e-14);
    CHECK_COMPLEX_NEAR(result.psi0_code_tetrad,
                       factor * baseline.psi0_code_tetrad, 3.0e-13);
  }
}

TEST_CASE("linear Psi0 inner Jet tangents use the explicit second stage") {
  constexpr double epsilon = 2.0e-6;
  for (const auto& fixture :
       teuk::test::plus2_linear_psi0::coordinate_oracle_fixtures()) {
    const double sine = std::sin(fixture.theta);
    const double cosine = std::cos(fixture.theta);
    const auto baseline = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine, fixture.stage,
        fixture.derivatives);
    auto plus_stage = fixture.stage;
    plus_stage.h = shifted(fixture.stage.h, fixture.stage.h_T, epsilon);
    plus_stage.h_T =
        shifted(fixture.stage.h_T, fixture.stage.h_TT, epsilon);
    auto minus_stage = fixture.stage;
    minus_stage.h = shifted(fixture.stage.h, fixture.stage.h_T, -epsilon);
    minus_stage.h_T =
        shifted(fixture.stage.h_T, fixture.stage.h_TT, -epsilon);
    auto plus_derivatives = fixture.derivatives;
    plus_derivatives.h_R = shifted(fixture.derivatives.h_R,
                                   fixture.derivatives.h_TR, epsilon);
    plus_derivatives.h_theta = shifted(fixture.derivatives.h_theta,
                                       fixture.derivatives.h_Ttheta, epsilon);
    plus_derivatives.h_phi = shifted(fixture.derivatives.h_phi,
                                     fixture.derivatives.h_Tphi, epsilon);
    auto minus_derivatives = fixture.derivatives;
    minus_derivatives.h_R = shifted(fixture.derivatives.h_R,
                                    fixture.derivatives.h_TR, -epsilon);
    minus_derivatives.h_theta = shifted(fixture.derivatives.h_theta,
                                        fixture.derivatives.h_Ttheta,
                                        -epsilon);
    minus_derivatives.h_phi = shifted(fixture.derivatives.h_phi,
                                      fixture.derivatives.h_Tphi, -epsilon);
    const auto plus = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine, plus_stage,
        plus_derivatives);
    const auto minus = teuk::evaluate_plus2_linear_psi0(
        fixture.parameters, fixture.radius, sine, cosine, minus_stage,
        minus_derivatives);
    CHECK_COMPLEX_NEAR((plus.sigma1 - minus.sigma1) / (2.0 * epsilon),
                       baseline.sigma1_T, 2.0e-10);
    CHECK_COMPLEX_NEAR((plus.kappa1 - minus.kappa1) / (2.0 * epsilon),
                       baseline.kappa1_T, 2.0e-10);
  }
}

TEST_CASE("linear Psi0 metric construction requires signed sharp partners") {
  const KC B_minus(0.3, -0.8);
  const KC C_minus(-0.4, 0.2);
  const KC U_plus(0.7, -0.1);
  const KC mu0(-0.6, 0.25);
  constexpr double radius = 0.42;
  const KC B_sharp = Kokkos::conj(B_minus);
  const KC C_sharp = Kokkos::conj(C_minus);
  const auto metric = teuk::plus2_org_metric_from_reconstruction(
      radius, mu0, B_sharp, C_sharp, U_plus);
  CHECK_COMPLEX_NEAR(metric.h_mm, radius * Kokkos::conj(B_minus), 1.0e-15);
  CHECK_COMPLEX_NEAR(metric.h_lm,
                     radius * radius * Kokkos::conj(C_minus), 1.0e-15);
  CHECK_COMPLEX_NEAR(metric.h_ll, radius * radius * U_plus / mu0, 1.0e-15);
  CHECK(Kokkos::abs(metric.h_mm - radius * Kokkos::conj(KC(0.9, 0.1))) >
        0.1);

  using Jet = teuk::Jet1<KC>;
  const auto metric_jet = teuk::plus2_org_metric_from_reconstruction(
      radius, Jet(mu0, KC(0.1, -0.2)), Jet(B_sharp, KC(-0.3, 0.4)),
      Jet(C_sharp, KC(0.2, 0.5)), Jet(U_plus, KC(-0.6, 0.3)));
  const double step = 1.0e-7;
  const auto shifted_metric = teuk::plus2_org_metric_from_reconstruction(
      radius, mu0 + step * KC(0.1, -0.2),
      B_sharp + step * KC(-0.3, 0.4), C_sharp + step * KC(0.2, 0.5),
      U_plus + step * KC(-0.6, 0.3));
  CHECK_COMPLEX_NEAR((shifted_metric.h_ll - metric_jet.h_ll.value) / step,
                     metric_jet.h_ll.dt, 2.0e-8);
  CHECK_COMPLEX_NEAR((shifted_metric.h_lm - metric_jet.h_lm.value) / step,
                     metric_jet.h_lm.dt, 2.0e-9);
  CHECK_COMPLEX_NEAR((shifted_metric.h_mm - metric_jet.h_mm.value) / step,
                     metric_jet.h_mm.dt, 2.0e-9);
}

TEST_CASE("linear Psi0 regularization fails closed and uses exact scri limit") {
  constexpr double length = 2.3;
  constexpr double spin = 0.999;
  constexpr double cosine = -0.37;
  const KC z_plus(0.42, -0.19);
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const KC psi0_over_radius5 = z_plus / (length4 * length4);
  const auto missing = teuk::regularize_plus2_linear_psi0(
      KC(), 0.0, cosine, spin, length);
  CHECK(!missing.valid);
  const auto scri = teuk::regularize_plus2_linear_psi0(
      KC(), 0.0, cosine, spin, length, true, psi0_over_radius5);
  CHECK(scri.valid);
  CHECK_COMPLEX_NEAR(scri.z_plus, z_plus, 2.0e-15);
  CHECK(!teuk::regularize_plus2_linear_psi0(
             KC(), -0.1, cosine, spin, length, true, psi0_over_radius5)
             .valid);

  for (const double radius : {1.0e-2, 2.0e-4, 3.0e-6}) {
    const KC psi0 = teuk::plus2_code_tetrad_scaling(
                        radius, cosine, spin, length) *
                    z_plus;
    const auto interior = teuk::regularize_plus2_linear_psi0(
        psi0, radius, cosine, spin, length);
    CHECK(interior.valid);
    CHECK_COMPLEX_NEAR(interior.z_plus, z_plus, 3.0e-15);
  }

  const teuk::TeukolskyParameters parameters{1.0, spin, length};
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const KC horizon_psi0 = teuk::plus2_code_tetrad_scaling(
                              horizon, cosine, spin, length) *
                          z_plus;
  const auto at_horizon = teuk::regularize_plus2_linear_psi0(
      horizon_psi0, horizon, cosine, spin, length);
  CHECK(at_horizon.valid);
  CHECK_COMPLEX_NEAR(at_horizon.z_plus, z_plus, 3.0e-15);
}

TEST_CASE("linear Psi0 modal azimuthal slots are exact") {
  teuk::Plus2OrgMetricStage stage{
      {KC(0.2, -0.3), KC(-0.4, 0.5), KC(0.6, 0.7)},
      {KC(-0.8, 0.1), KC(0.9, -0.2), KC(-0.3, -0.5)},
      {},
  };
  teuk::Plus2OrgMetricDerivativeSlots derivatives;
  derivatives.h_R = {KC(0.4, 0.2), KC(-0.7, 0.6), KC(0.3, -0.9)};
  derivatives.h_theta = {KC(-0.1, 0.8), KC(0.5, 0.4), KC(-0.6, 0.2)};
  constexpr int mode = -3;
  const KC im(0.0, mode);
  teuk::plus2_fill_modal_azimuthal_derivatives(mode, stage, derivatives);
  CHECK_COMPLEX_NEAR(derivatives.h_phi.h_lm, im * stage.h.h_lm, 1.0e-15);
  CHECK_COMPLEX_NEAR(derivatives.h_Tphi.h_mm, im * stage.h_T.h_mm,
                     1.0e-15);
  CHECK_COMPLEX_NEAR(derivatives.h_Rphi.h_ll, im * derivatives.h_R.h_ll,
                     1.0e-15);
  CHECK_COMPLEX_NEAR(derivatives.h_thetaphi.h_lm,
                     im * derivatives.h_theta.h_lm, 1.0e-15);
  CHECK_COMPLEX_NEAR(derivatives.h_phiphi.h_mm,
                     -9.0 * stage.h.h_mm, 1.0e-15);
}

TEST_CASE("linear Psi0 point operator has device parity and no allocation") {
  const auto fixture =
      teuk::test::plus2_linear_psi0::coordinate_oracle_fixtures()[3];
  const double sine = std::sin(fixture.theta);
  const double cosine = std::cos(fixture.theta);
  const auto host_result = teuk::evaluate_plus2_linear_psi0(
      fixture.parameters, fixture.radius, sine, cosine, fixture.stage,
      fixture.derivatives);
  Kokkos::View<teuk::Plus2LinearPsi0Point*> output("linear_psi0_output", 1);
  linear_psi0_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_linear_psi0_allocation);
  Kokkos::parallel_for(
      "linear_psi0_device_parity", 1, KOKKOS_LAMBDA(const int) {
        output(0) = teuk::evaluate_plus2_linear_psi0(
            fixture.parameters, fixture.radius, sine, cosine, fixture.stage,
            fixture.derivatives);
      });
  Kokkos::fence();
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(linear_psi0_allocations == 0);
  const auto copy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        output);
  CHECK_COMPLEX_NEAR(copy(0).sigma1, host_result.sigma1, 2.0e-15);
  CHECK_COMPLEX_NEAR(copy(0).sigma1_T, host_result.sigma1_T, 2.0e-15);
  CHECK_COMPLEX_NEAR(copy(0).kappa1, host_result.kappa1, 2.0e-15);
  CHECK_COMPLEX_NEAR(copy(0).kappa1_T, host_result.kappa1_T, 2.0e-15);
  CHECK_COMPLEX_NEAR(copy(0).psi0_code_tetrad,
                     host_result.psi0_code_tetrad, 2.0e-15);
}
