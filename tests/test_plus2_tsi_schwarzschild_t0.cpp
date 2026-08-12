#include "test_harness.hpp"

#include <cmath>
#include <complex>
#include <cstring>

#include "plus2_tsi_schwarzschild_t0_fixtures.hpp"
#include "teuk/plus2_linear_psi0.hpp"

namespace {

using KC = teuk::Complex;
using Fields = teuk::Plus2OrgMetricFields;

KC kc(const teuk::test::plus2_tsi::C& value) {
  return {value.real, value.imag};
}

Fields separated_values(
    const teuk::test::plus2_tsi::T0Fixture& fixture,
    const int radial_derivative, const int angular_derivative) {
  Fields result;
  KC* components[] = {&result.h_ll, &result.h_lm, &result.h_mm};
  const double radial_factor = radial_derivative == 2 ? 2.0 : 1.0;
  const double angular_factor = angular_derivative == 2 ? 2.0 : 1.0;
  for (std::size_t component = 0; component < 3; ++component) {
    *components[component] =
        radial_factor * angular_factor *
        kc(fixture.metric[component].radial[radial_derivative]) *
        kc(fixture.metric[component].angular[angular_derivative]);
  }
  return result;
}

Fields scaled(const KC factor, const Fields& value) {
  return {factor * value.h_ll, factor * value.h_lm, factor * value.h_mm};
}

}  // namespace

TEST_CASE("normalized Schwarzschild TSI metric reproduces both T0 modes") {
  const teuk::KerrParameters parameters{1.0, 0.0, 1.0};
  std::array<int, 3> level_counts{};
  for (const auto& fixture :
       teuk::test::plus2_tsi::schwarzschild_t0_fixtures()) {
    const int level = std::strcmp(fixture.level, "coarse") == 0 ? 0
                      : std::strcmp(fixture.level, "medium") == 0 ? 1
                                                                  : 2;
    ++level_counts[level];
    const KC minus_i_omega(0.0, -fixture.omega);
    teuk::Plus2OrgMetricStage stage;
    stage.h = separated_values(fixture, 0, 0);
    stage.h_T = scaled(minus_i_omega, stage.h);
    stage.h_TT = scaled(-fixture.omega * fixture.omega, stage.h);

    teuk::Plus2OrgMetricDerivativeSlots derivatives;
    derivatives.h_R = separated_values(fixture, 1, 0);
    derivatives.h_RR = separated_values(fixture, 2, 0);
    derivatives.h_theta = separated_values(fixture, 0, 1);
    derivatives.h_thetatheta = separated_values(fixture, 0, 2);
    derivatives.h_Rtheta = separated_values(fixture, 1, 1);
    derivatives.h_TR = scaled(minus_i_omega, derivatives.h_R);
    derivatives.h_Ttheta = scaled(minus_i_omega, derivatives.h_theta);
    teuk::plus2_fill_modal_azimuthal_derivatives(fixture.m, stage,
                                                 derivatives);

    const auto actual = teuk::evaluate_plus2_linear_psi0(
        parameters, fixture.radius, std::sin(fixture.theta),
        std::cos(fixture.theta), stage, derivatives);
    // The recorded maximum across all twelve cases is 1.42e-15.  The 2e-14
    // absolute gate leaves a factor fourteen for binary64 serialization and
    // compiler reassociation; it is not based on the looser ODE error bound.
    CHECK_COMPLEX_NEAR(actual.psi0_code_tetrad, kc(fixture.expected_psi0_code),
                       2.0e-14);
  }
  const std::array<int, 3> expected_counts{4, 4, 4};
  CHECK(level_counts == expected_counts);
}
