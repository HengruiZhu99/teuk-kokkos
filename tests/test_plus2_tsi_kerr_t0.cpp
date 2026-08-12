#include "test_harness.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>

#include "plus2_tsi_kerr_t0_fixtures.hpp"
#include "teuk/plus2_linear_psi0.hpp"

namespace fixture = teuk::test::plus2_tsi_kerr_t0;

namespace {

using KC = teuk::Complex;
using Fields = teuk::Plus2OrgMetricFields;

KC kc(const fixture::C& value) { return {value.real, value.imag}; }

Fields radial_values(const fixture::Fixture& value, const int derivative) {
  Fields result;
  KC* components[] = {&result.h_ll, &result.h_lm, &result.h_mm};
  const double factor = derivative == 2 ? 2.0 : 1.0;
  for (std::size_t component = 0; component < 3; ++component) {
    *components[component] =
        factor * kc(value.metric[component].radial[derivative]);
  }
  return result;
}

Fields angular_values(const fixture::Fixture& value, const int derivative) {
  Fields result;
  KC* components[] = {&result.h_ll, &result.h_lm, &result.h_mm};
  const double factor = derivative == 2 ? 2.0 : 1.0;
  for (std::size_t component = 0; component < 3; ++component) {
    *components[component] =
        factor * kc(value.metric[component].angular[derivative]);
  }
  return result;
}

Fields mixed_values(const fixture::Fixture& value) {
  return {kc(value.metric[0].mixed_Rtheta),
          kc(value.metric[1].mixed_Rtheta),
          kc(value.metric[2].mixed_Rtheta)};
}

Fields scaled(const KC factor, const Fields& value) {
  return {factor * value.h_ll, factor * value.h_lm, factor * value.h_mm};
}

}  // namespace

TEST_CASE("normalized moderate Kerr ORG metric reproduces both T0 sectors") {
  const teuk::KerrParameters parameters{1.0, 0.6, 1.0};
  std::array<int, 3> level_counts{};
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  for (const auto& value : fixture::fixtures()) {
    const int level = std::strcmp(value.level, "coarse") == 0 ? 0
                      : std::strcmp(value.level, "medium") == 0 ? 1
                                                                  : 2;
    ++level_counts[level];
    const KC minus_i_omega(0.0, -value.omega);
    teuk::Plus2OrgMetricStage stage;
    stage.h = radial_values(value, 0);
    stage.h_T = scaled(minus_i_omega, stage.h);
    stage.h_TT = scaled(-value.omega * value.omega, stage.h);

    teuk::Plus2OrgMetricDerivativeSlots derivatives;
    derivatives.h_R = radial_values(value, 1);
    derivatives.h_RR = radial_values(value, 2);
    derivatives.h_theta = angular_values(value, 1);
    derivatives.h_thetatheta = angular_values(value, 2);
    derivatives.h_Rtheta = mixed_values(value);
    derivatives.h_TR = scaled(minus_i_omega, derivatives.h_R);
    derivatives.h_Ttheta = scaled(minus_i_omega, derivatives.h_theta);
    teuk::plus2_fill_modal_azimuthal_derivatives(value.m, stage, derivatives);

    const auto actual = teuk::evaluate_plus2_linear_psi0(
        parameters, value.radius, std::sin(value.theta), std::cos(value.theta),
        stage, derivatives);
    const KC expected = kc(value.expected_psi0_code);
    const double error = Kokkos::abs(actual.psi0_code_tetrad - expected);
    maximum_absolute_error = std::max(maximum_absolute_error, error);
    maximum_relative_error =
        std::max(maximum_relative_error,
                 error / std::max(Kokkos::abs(expected), 1.0e-300));
    CHECK_COMPLEX_NEAR(actual.psi0_code_tetrad, expected, 3.0e-11);
    CHECK(error / std::max(Kokkos::abs(expected), 1.0e-300) < 4.0e-10);
  }
  const std::array<int, 3> expected_counts{4, 4, 4};
  CHECK(level_counts == expected_counts);
  std::cout << "moderate Kerr normalized T0 maximum absolute/relative errors "
            << maximum_absolute_error << ' ' << maximum_relative_error << '\n';
}

TEST_CASE("moderate Kerr T0 fixture records exact source provenance") {
  CHECK(std::strlen(fixture::generator_sha256) == 64);
  CHECK(std::strlen(fixture::separated_solver_sha256) == 64);
  CHECK(std::strlen(fixture::schwarzschild_generator_sha256) == 64);
  CHECK(std::strlen(fixture::berens_arxiv_tex_sha256) == 64);
  CHECK(std::strlen(fixture::berens_supplement_commit) == 40);
  CHECK(std::strlen(fixture::berens_example_nb_sha256) == 64);
  CHECK(std::strlen(fixture::ripley_arxiv_tex_sha256) == 64);
}
