#include "test_harness.hpp"

#include "teuk/ghp.hpp"
#include "teuk/reconstruction.hpp"

namespace {

using teuk::Complex;

TEST_CASE("all seven rescaled reconstruction equations match the reference") {
  const double radius = 0.37;
  const teuk::KerrBackgroundPoint bg{Complex(-0.7, 0.2),
                                      Complex(0.04, 0.09),
                                      Complex(-0.03, -0.08),
                                      Complex(-0.4, 0.3)};
  const teuk::ReconstructionFields f{
      Complex(0.2, 0.5),   Complex(-0.3, 0.7), Complex(0.9, -0.1),
      Complex(-0.6, -0.2), Complex(0.1, 0.8),  Complex(0.4, -0.9),
      Complex(-0.5, 0.3),  Complex(0.7, 0.6),  Complex(-0.2, -0.1),
      Complex(0.8, -0.4),  Complex(-0.7, 0.2)};
  const teuk::ReconstructionAngularDerivatives a{
      Complex(0.11, -0.17), Complex(-0.21, 0.13), Complex(0.31, 0.27),
      Complex(-0.41, -0.07), Complex(0.51, 0.37), Complex(-0.61, 0.47)};
  const auto rhs = teuk::reconstruction_delta_rhs(radius, bg, f, a);

  const Complex mu_bar = Kokkos::conj(bg.mu0);
  const Complex pi_bar = Kokkos::conj(bg.pi0);
  const Complex tau_bar = Kokkos::conj(bg.tau0);
  const double r2 = radius * radius;
  const Complex expected_G =
      -4.0 * radius * bg.mu0 * f.G + a.eth1_F - radius * bg.tau0 * f.F;
  const Complex expected_Lambda =
      -radius * (bg.mu0 + mu_bar) * f.Lambda - f.F;
  const Complex expected_H =
      -3.0 * radius * bg.mu0 * f.H + a.eth2_G -
      2.0 * radius * bg.tau0 * f.G;
  const Complex expected_B =
      radius * (bg.mu0 - mu_bar) * f.B - 2.0 * f.Lambda;
  const Complex expected_Pi =
      -f.G - radius * (pi_bar + bg.tau0) * f.Lambda +
      0.5 * r2 * bg.mu0 * (pi_bar + bg.tau0) * f.B;
  const Complex expected_C =
      -radius * mu_bar * f.C - 2.0 * f.Pi - radius * bg.tau0 * f.B;
  const Complex expected_U =
      -radius * mu_bar * f.U - radius * bg.mu0 * a.eth2_C -
      r2 * bg.mu0 * (pi_bar + 2.0 * bg.tau0) * f.C - 2.0 * a.eth2_Pi -
      2.0 * radius * pi_bar * f.Pi - 2.0 * f.H -
      2.0 * radius * bg.pi0 * f.Pi_sharp -
      radius * bg.pi0 * a.ethprime1_B_sharp +
      r2 * bg.pi0 * bg.pi0 * f.B_sharp +
      radius * bg.mu0 * a.ethprime2_C_sharp +
      r2 * (-3.0 * bg.mu0 * bg.pi0 + 2.0 * mu_bar * bg.pi0 -
            2.0 * bg.mu0 * tau_bar) *
          f.C_sharp;

  CHECK_COMPLEX_NEAR(rhs.G, expected_G, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.Lambda, expected_Lambda, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.H, expected_H, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.B, expected_B, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.Pi, expected_Pi, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.C, expected_C, 1.0e-14);
  CHECK_COMPLEX_NEAR(rhs.U, expected_U, 1.0e-14);
}

TEST_CASE("reconstruction time solve inverts Delta_n point formula") {
  const Complex value(0.4, -0.6);
  const Complex radial_derivative(-0.2, 0.9);
  const Complex expected_dt(0.7, 0.1);
  const int falloff = 2;
  const double radius = 0.44;
  const double mass = 1.0;
  const double length = 1.35;
  const Complex rhs = teuk::delta_n_point(value, expected_dt,
                                           radial_derivative, falloff, radius,
                                           mass, length);
  const Complex actual_dt = teuk::reconstruction_time_derivative(
      value, radial_derivative, rhs, falloff, radius, mass, length);
  CHECK_COMPLEX_NEAR(actual_dt, expected_dt, 1.0e-14);
}

TEST_CASE("sharp reconstruction inputs are explicit independent values") {
  teuk::ReconstructionFields f{};
  f.Pi = Complex(1.0, 2.0);
  f.Pi_sharp = Complex(-3.0, 4.0);
  CHECK(Kokkos::abs(f.Pi_sharp - Kokkos::conj(f.Pi)) > 1.0);
}

}  // namespace
