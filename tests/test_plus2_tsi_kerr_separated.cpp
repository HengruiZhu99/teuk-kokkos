#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

#include "plus2_tsi_kerr_separated_fixtures.hpp"

namespace fixture = teuk::test::plus2_tsi_kerr;

TEST_CASE("generated moderate Kerr TSI fixture pins normalized factors") {
  CHECK_NEAR(fixture::mass, 1.0, 0.0);
  CHECK_NEAR(fixture::spin, 0.6, 0.0);
  CHECK_NEAR(fixture::omega, 0.2, 0.0);
  CHECK(fixture::ell == 2);
  CHECK(fixture::m == 2);
  CHECK(std::strlen(fixture::generator_sha256) == 64);
  CHECK(std::strlen(fixture::berens_arxiv_tex_sha256) == 64);
  CHECK(std::strlen(fixture::berens_example_nb_sha256) == 64);
  CHECK(std::strlen(fixture::berens_supplement_commit) == 40);

  for (const auto& level : fixture::levels) {
    CHECK_NEAR(level.lambda_minus, level.lambda_plus + 4.0, 2.0e-10);
    CHECK_NEAR(level.d_hat * level.d_hat_prime, level.d_product, 2.0e-11);
    const std::complex<double> c_hat(level.c_hat.real, level.c_hat.imag);
    const std::complex<double> c_hat_prime(level.c_hat_prime.real,
                                           level.c_hat_prime.imag);
    CHECK(std::abs(c_hat * c_hat_prime - level.c_product) < 2.0e-11);
    const std::complex<double> sharp_to_same(level.sharp_to_same.real,
                                              level.sharp_to_same.imag);
    const auto expected_ratio = 0.5 * std::complex<double>(0.0, fixture::omega) *
                                c_hat_prime / std::conj(c_hat_prime);
    CHECK(std::abs(sharp_to_same - expected_ratio) < 2.0e-15);
    CHECK(std::abs(sharp_to_same -
                   0.5 * std::complex<double>(0.0, fixture::omega)) > 1.0e-3);
  }
}

TEST_CASE("moderate Kerr hatted modes and TSI identities are converged") {
  const auto& medium = fixture::levels[1];
  const auto& fine = fixture::levels[2];
  CHECK(fine.angular_mode_change < medium.angular_mode_change);
  CHECK(fine.radial_mode_change < medium.radial_mode_change);
  CHECK(fine.angular_mode_change < 2.0e-10);
  CHECK(fine.radial_mode_change < 2.0e-10);
  CHECK(std::max(fine.angular_forward, fine.angular_backward) < 2.0e-8);
  CHECK(*std::max_element(fine.radial_residuals.begin(),
                          fine.radial_residuals.end()) < 5.0e-7);
  CHECK(fine.signed_radial < 2.0e-8);
  CHECK(fine.signed_angular < 2.0e-8);
}
