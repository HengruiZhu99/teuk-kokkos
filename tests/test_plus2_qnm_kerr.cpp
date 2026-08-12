#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

#include "plus2_qnm_kerr_fixtures.hpp"
#include "teuk/plus2_field.hpp"
#include "teuk/teukolsky.hpp"

namespace fixture = teuk::test::plus2_qnm_kerr;

namespace {

using KC = teuk::Complex;

KC kc(const fixture::C value) { return {value.real, value.imag}; }

double relative_difference(const KC actual, const KC expected) {
  return Kokkos::abs(actual - expected) /
         std::max(Kokkos::abs(expected), 1.0e-300);
}

double fixture_difference(const fixture::Fixture& left,
                          const fixture::Fixture& right) {
  double result = relative_difference(kc(left.lambda_plus),
                                      kc(right.lambda_plus));
  result = std::max(
      result, relative_difference(kc(left.psi0_code), kc(right.psi0_code)));
  result = std::max(
      result, relative_difference(kc(left.angular_laplacian),
                                  kc(right.angular_laplacian)));
  for (std::size_t derivative = 0; derivative < 3; ++derivative) {
    result = std::max(
        result,
        relative_difference(kc(left.stored[derivative]),
                            kc(right.stored[derivative])));
  }
  return result;
}

}  // namespace

TEST_CASE("normalized moderate Kerr spin plus2 QNM satisfies stored-field operator") {
  for (const auto& value : fixture::fixtures()) {
    const KC omega = kc(value.omega);
    const KC rate = KC(0.0, -1.0) * omega;
    const KC z = kc(value.stored[0]);
    const KC z_r = kc(value.stored[1]);
    const KC z_rr = 2.0 * kc(value.stored[2]);
    const teuk::TeukolskyParameters parameters{
        1.0, 0.6, 1.0, 2, value.m, 0.0};
    const auto coefficients =
        teuk::teukolsky_coefficients(parameters, value.radius, value.theta);
    const teuk::TeukolskyState state{
        coefficients.time * rate * z -
            2.0 * coefficients.radial_advection * z_r +
            coefficients.definition * z,
        z_r,
        z};

    CHECK(relative_difference(teuk::teukolsky_psi_rhs(coefficients, state),
                              rate * z) < 8.0e-14);
    const KC p_rhs = teuk::teukolsky_p_rhs(
        coefficients, state, z_rr, kc(value.angular_laplacian), KC(0.0));
    CHECK(relative_difference(p_rhs, rate * state.P) < 8.0e-13);
    CHECK(relative_difference(
              teuk::teukolsky_q_rhs(rate * z_r, KC(0.0), 0.0),
              rate * state.Q) < 3.0e-15);

    const KC raw = teuk::plus2_code_tetrad_psi0(
        z, parameters, value.radius, std::cos(value.theta));
    CHECK(relative_difference(raw, kc(value.psi0_code)) < 8.0e-14);

    // Omitting the Kerr angular action must not accidentally satisfy the
    // homogeneous compact equation.
    const KC wrong_p_rhs = teuk::teukolsky_p_rhs(
        coefficients, state, z_rr, KC(0.0), KC(0.0));
    CHECK(relative_difference(wrong_p_rhs, rate * state.P) > 1.0e-3);
    CHECK(value.wronskian < 2.0e-10);
  }
}

TEST_CASE("normalized moderate Kerr spin plus2 QNM fixture converges") {
  const auto values = fixture::fixtures();
  double coarse_medium = 0.0;
  double medium_fine = 0.0;
  for (std::size_t sector_point = 0; sector_point < 4; ++sector_point) {
    coarse_medium = std::max(
        coarse_medium,
        fixture_difference(values[sector_point], values[4 + sector_point]));
    medium_fine = std::max(
        medium_fine,
        fixture_difference(values[4 + sector_point],
                           values[8 + sector_point]));
  }
  CHECK(medium_fine < coarse_medium);
  CHECK(medium_fine < 3.0e-8);

  for (std::size_t level = 0; level < 3; ++level) {
    for (std::size_t point = 0; point < 2; ++point) {
      const auto& mode = values[4 * level + point];
      const auto& sharp = values[4 * level + 2 + point];
      CHECK(mode.point == sharp.point);
      CHECK(mode.m == -sharp.m);
      CHECK_COMPLEX_NEAR(kc(sharp.omega), -Kokkos::conj(kc(mode.omega)),
                         2.0e-15);
      CHECK_COMPLEX_NEAR(kc(sharp.lambda_plus),
                         Kokkos::conj(kc(mode.lambda_plus)), 3.0e-11);
    }
  }
}

TEST_CASE("normalized moderate Kerr spin plus2 QNM records independent provenance") {
  CHECK(std::strcmp(fixture::qnm_version, "0.4.4") == 0);
  CHECK(std::strlen(fixture::qnm_wheel_sha256) == 64);
  CHECK(std::strlen(fixture::generator_sha256) == 64);
  CHECK(std::strlen(fixture::separated_solver_sha256) == 64);
  CHECK(std::strlen(fixture::transform_solver_sha256) == 64);
  CHECK(fixture::wrong_qnm_A_wronskian > 1.0);

  const auto values = fixture::fixtures();
  CHECK_COMPLEX_NEAR(kc(values[0].omega),
                     KC(0.4940447817813845, -0.0837652021610416), 0.0);
  CHECK_COMPLEX_NEAR(kc(values[0].lambda_plus),
                     KC(-1.9549779674091456, 0.32793056263873765),
                     2.0e-10);
  // qnm's A is not Berens' lambda_plus; the deliberately wrong direct use is
  // separated by an O(10) complex angular Wronskian residual.
  CHECK(Kokkos::abs(kc(values[0].lambda_plus) -
                    KC(-0.8546134005662634, 0.15669038538845606)) > 1.0);
}
