#include "test_harness.hpp"

#include <cmath>
#include <complex>

#include "teuk/background.hpp"
#include "teuk/ghp.hpp"

namespace {

using teuk::Complex;

TEST_CASE("Schwarzschild background coefficients have the exact limit") {
  const teuk::KerrParameters parameters{1.25, 0.0, 2.0};
  const auto background =
      teuk::kerr_background_point(parameters, 0.3, 0.4, std::sqrt(0.84));

  CHECK_COMPLEX_NEAR(background.mu0, Complex(-0.25, 0.0), 1.0e-14);
  CHECK_COMPLEX_NEAR(background.tau0, Complex(0.0, 0.0), 1.0e-14);
  CHECK_COMPLEX_NEAR(background.pi0, Complex(0.0, 0.0), 1.0e-14);
  CHECK_COMPLEX_NEAR(background.psi20, Complex(-1.25 / 64.0, 0.0),
                     1.0e-14);
}

TEST_CASE("rotating equatorial background exposes the Kerr connection") {
  const double spin = 0.8;
  const double length = 1.7;
  const double length4 = length * length * length * length;
  const teuk::KerrParameters parameters{1.0, spin, length};
  const auto background =
      teuk::kerr_background_point(parameters, 0.45, 0.0, 1.0);

  const Complex expected_connection(0.0,
      3.0 * spin / (std::sqrt(2.0) * length4));
  CHECK_COMPLEX_NEAR(Kokkos::conj(background.pi0) + 2.0 * background.tau0,
                     expected_connection, 1.0e-14);
  CHECK(Kokkos::abs(expected_connection) > 0.0);
}

TEST_CASE("background formula runs through the active Kokkos execution space") {
  Kokkos::View<Complex*> device_value("background value", 1);
  const teuk::KerrParameters parameters{1.0, 0.63, 1.2};
  Kokkos::parallel_for(
      "evaluate Kerr background", 1, KOKKOS_LAMBDA(const int) {
        device_value(0) =
            teuk::kerr_background_point(parameters, 0.31, 0.2, 0.7).psi20;
      });
  const auto host_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), device_value);
  const auto scalar_value =
      teuk::kerr_background_point(parameters, 0.31, 0.2, 0.7).psi20;
  CHECK_COMPLEX_NEAR(host_value(0), scalar_value, 1.0e-14);
}

TEST_CASE("rescaled Delta operator matches its coordinate point formula") {
  const Complex value(0.7, -0.2);
  const Complex dt_value(-0.1, 0.4);
  const Complex dr_value(0.3, 0.8);
  const double radius = 0.35;
  const double mass = 1.0;
  const double length = 1.4;
  const double length2 = length * length;
  const Complex expected =
      (2.0 + 4.0 * mass * radius / length2) * dt_value +
      (radius / length2) * (radius * dr_value + 3.0 * value);
  CHECK_COMPLEX_NEAR(teuk::delta_n_point(value, dt_value, dr_value, 3,
                                         radius, mass, length),
                     expected, 1.0e-14);
}

TEST_CASE("rescaled eth and eth-prime reduce correctly in Schwarzschild") {
  const Complex value(0.2, 0.6);
  const Complex dt_value(-0.4, 0.1);
  const Complex raised_value(0.9, -0.3);
  const Complex lowered_value(-0.7, 0.5);
  const double length = 1.3;
  const double denominator = std::sqrt(2.0) * length * length;

  const Complex eth = teuk::eth_n_point(value, dt_value, raised_value, -2, -2,
                                         0.4, 0.8, 0.6, 0.0, length);
  const Complex ethprime = teuk::ethprime_n_point(
      value, dt_value, lowered_value, -2, -2, 0.4, 0.8, 0.6, 0.0, length);
  CHECK_COMPLEX_NEAR(eth, raised_value / denominator, 1.0e-14);
  CHECK_COMPLEX_NEAR(ethprime, lowered_value / denominator, 1.0e-14);
}

TEST_CASE("rescaled eth point formula includes spin-boost connection") {
  const Complex value(0.3, -0.8);
  const Complex dt_value(0.4, 0.2);
  const Complex raised_value(-0.1, 0.7);
  const double radius = 0.27;
  const double sin_theta = 0.8;
  const double cos_theta = 0.6;
  const double spin = 0.71;
  const double length = 1.1;
  const int spin_weight = -1;
  const int boost_weight = 0;
  const std::complex<double> i(0.0, 1.0);
  const std::complex<double> z(value.real(), value.imag());
  const std::complex<double> zt(dt_value.real(), dt_value.imag());
  const std::complex<double> zr(raised_value.real(), raised_value.imag());
  const std::complex<double> denominator(length * length,
                                         -spin * radius * cos_theta);
  const auto expected =
      (-i * spin * sin_theta * zt + zr) /
          (std::sqrt(2.0) * denominator) -
      i * static_cast<double>(spin_weight + boost_weight) * spin * radius *
          sin_theta * z /
          (std::sqrt(2.0) * denominator * denominator);
  const Complex actual = teuk::eth_n_point(
      value, dt_value, raised_value, spin_weight, boost_weight, radius,
      sin_theta, cos_theta, spin, length);
  CHECK_COMPLEX_NEAR(actual, Complex(expected.real(), expected.imag()),
                     1.0e-14);
}

}  // namespace
