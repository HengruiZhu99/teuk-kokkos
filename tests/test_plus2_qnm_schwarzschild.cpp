#include "test_harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>

#include "plus2_qnm_schwarzschild_fixtures.hpp"
#include "teuk/plus2_field.hpp"
#include "teuk/teukolsky.hpp"

namespace fixture = teuk::test::plus2_qnm_schwarzschild;

namespace {

using KC = teuk::Complex;

KC kc(const fixture::C value) { return {value.real, value.imag}; }

double relative_difference(const KC actual, const KC expected) {
  return Kokkos::abs(actual - expected) /
         std::max(Kokkos::abs(expected), 1.0e-300);
}

double angular_factor(const double theta, const int m) {
  const double cosine = std::cos(theta);
  return m == 2 ? (1.0 - cosine) * (1.0 - cosine)
                : (1.0 + cosine) * (1.0 + cosine);
}

}  // namespace

TEST_CASE("normalized Schwarzschild spin plus2 QNM satisfies stored-field operator") {
  for (const auto& value : fixture::fixtures()) {
    const KC omega = kc(value.omega);
    const KC rate = KC(0.0, -1.0) * omega;
    const KC z = kc(value.stored[0]);
    const KC z_r = kc(value.stored[1]);
    const KC z_rr = 2.0 * kc(value.stored[2]);
    const teuk::TeukolskyParameters parameters{
        1.0, 0.0, 1.0, 2, value.m, 0.0};
    const auto coefficients =
        teuk::teukolsky_coefficients(parameters, value.radius, value.theta);

    const teuk::TeukolskyState state{
        coefficients.time * rate * z -
            2.0 * coefficients.radial_advection * z_r +
            coefficients.definition * z,
        z_r,
        z};
    CHECK_COMPLEX_NEAR(teuk::teukolsky_psi_rhs(coefficients, state), rate * z,
                       4.0e-15);

    // For s=+2, ell=2 in Schwarzschild, the repository's lower-after-raise
    // angular eigenvalue is -(ell-s)(ell+s+1)=0.
    const KC p_rhs = teuk::teukolsky_p_rhs(
        coefficients, state, z_rr, KC(0.0, 0.0), KC(0.0, 0.0));
    CHECK(relative_difference(p_rhs, rate * state.P) < 5.0e-14);
    CHECK_COMPLEX_NEAR(
        teuk::teukolsky_q_rhs(rate * z_r, KC(0.0, 0.0), 0.0),
        rate * state.Q, 2.0e-15);

    const KC raw = teuk::plus2_code_tetrad_psi0(
        z, parameters, value.radius, std::cos(value.theta));
    CHECK(relative_difference(raw, kc(value.psi0_code)) < 3.0e-14);
  }
}

TEST_CASE("normalized Schwarzschild spin plus2 QNM fixture converges and is sharp paired") {
  const auto values = fixture::fixtures();
  double coarse_medium = 0.0;
  double medium_fine = 0.0;
  for (std::size_t sector_point = 0; sector_point < 4; ++sector_point) {
    const auto compare = [&](const std::size_t left,
                             const std::size_t right) {
      double result = relative_difference(kc(values[left].psi0_code),
                                          kc(values[right].psi0_code));
      for (std::size_t derivative = 0; derivative < 3; ++derivative) {
        result = std::max(
            result, relative_difference(kc(values[left].stored[derivative]),
                                        kc(values[right].stored[derivative])));
      }
      return result;
    };
    coarse_medium =
        std::max(coarse_medium, compare(sector_point, 4 + sector_point));
    medium_fine =
        std::max(medium_fine, compare(4 + sector_point, 8 + sector_point));
  }
  CHECK(medium_fine < coarse_medium);
  CHECK(medium_fine < 2.0e-8);

  for (std::size_t level = 0; level < 3; ++level) {
    for (std::size_t point = 0; point < 2; ++point) {
      const auto& mode = values[4 * level + point];
      const auto& sharp = values[4 * level + 2 + point];
      CHECK(mode.point == sharp.point);
      CHECK(mode.m == -sharp.m);
      CHECK_COMPLEX_NEAR(kc(sharp.omega), -Kokkos::conj(kc(mode.omega)),
                         2.0e-15);
      const double mode_angular = angular_factor(mode.theta, mode.m);
      const double sharp_angular = angular_factor(sharp.theta, sharp.m);
      for (std::size_t derivative = 0; derivative < 3; ++derivative) {
        CHECK(relative_difference(kc(sharp.stored[derivative]) / sharp_angular,
                                  Kokkos::conj(kc(mode.stored[derivative]) /
                                               mode_angular)) < 4.0e-13);
      }
    }
  }
}

TEST_CASE("normalized Schwarzschild spin plus2 QNM records independent provenance") {
  CHECK(std::strcmp(fixture::qnm_version, "0.4.4") == 0);
  CHECK(std::strlen(fixture::qnm_wheel_sha256) == 64);
  CHECK(std::strlen(fixture::generator_sha256) == 64);
  CHECK(std::strlen(fixture::radial_solver_sha256) == 64);
  CHECK(std::strlen(fixture::berens_arxiv_tex_sha256) == 64);
  CHECK(std::strlen(fixture::berens_supplement_commit) == 40);
  CHECK(std::strlen(fixture::berens_example_nb_sha256) == 64);
  CHECK(std::strlen(fixture::ripley_arxiv_tex_sha256) == 64);
  const auto values = fixture::fixtures();
  CHECK_COMPLEX_NEAR(kc(values[0].omega),
                     KC(0.373671684418042, -0.08896231568893723), 0.0);
}
