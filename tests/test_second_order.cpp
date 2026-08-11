#include "test_harness.hpp"

#include <complex>

#include "teuk/second_order.hpp"

namespace {

using teuk::Complex;

std::complex<double> host_complex(const Complex& value) {
  return {value.real(), value.imag()};
}

TEST_CASE("Kokkos ordered-pair source agrees with transparent scalar oracle") {
  const double radius = 0.43;
  const teuk::KerrBackgroundPoint bg{
      Complex(-0.73, 0.18), Complex(0.02, 0.11),
      Complex(-0.04, -0.07), Complex(-0.6, 0.2)};
  const teuk::OrderedPairFields f{
      Complex(0.21, -0.17), Complex(-0.32, 0.27), Complex(0.43, 0.07),
      Complex(-0.54, -0.37), Complex(0.65, 0.47), Complex(-0.76, 0.57),
      Complex(0.87, -0.67), Complex(-0.98, -0.77), Complex(0.19, 0.87),
      Complex(-0.29, 0.97), Complex(0.39, -0.16), Complex(-0.49, -0.26),
      Complex(0.59, 0.36), Complex(-0.69, 0.46), Complex(0.79, -0.56),
      Complex(-0.89, -0.66), Complex(0.99, 0.76), Complex(-0.14, 0.86),
      Complex(0.24, -0.96)};
  const teuk::OrderedPairDerivatives d{
      Complex(0.12, 0.22),   Complex(-0.23, 0.33), Complex(0.34, -0.44),
      Complex(-0.45, -0.55), Complex(0.56, 0.66),  Complex(-0.67, 0.77),
      Complex(0.78, -0.88),  Complex(-0.89, -0.99), Complex(0.91, 0.13),
      Complex(-0.82, 0.24),  Complex(0.73, -0.35)};

  const auto actual = teuk::corrected_ordered_pair_source(radius, bg, f, d);
  const teuk::ReferenceOrderedPairFields rf{
      host_complex(f.F1), host_complex(f.G1), host_complex(f.Lambda1),
      host_complex(f.Pi1), host_complex(f.B1), host_complex(f.C1),
      host_complex(f.U1), host_complex(f.F2), host_complex(f.G2),
      host_complex(f.H2), host_complex(f.B2), host_complex(f.C2),
      host_complex(f.U2), host_complex(f.U2_sharp),
      host_complex(f.C2_sharp), host_complex(f.C1_sharp),
      host_complex(f.B1_sharp), host_complex(f.Pi2_sharp),
      host_complex(f.B2_sharp)};
  const teuk::ReferenceOrderedPairDerivatives rd{
      host_complex(d.delta1_F2), host_complex(d.delta3_U2),
      host_complex(d.eth2_C2), host_complex(d.ethprime2_C2_sharp),
      host_complex(d.eth1_B2), host_complex(d.delta2_C2),
      host_complex(d.delta2_G2), host_complex(d.eth2_G2),
      host_complex(d.ethprime1_F2), host_complex(d.delta2_C2_sharp),
      host_complex(d.ethprime1_B2_sharp)};
  const auto expected = teuk::corrected_ordered_pair_source_reference(
      radius, host_complex(bg.mu0), host_complex(bg.tau0),
      host_complex(bg.pi0), rf, rd);

  CHECK_COMPLEX_NEAR(actual.D, Complex(expected.D.real(), expected.D.imag()),
                     2.0e-14);
  CHECK_COMPLEX_NEAR(actual.T, Complex(expected.T.real(), expected.T.imag()),
                     2.0e-14);
}

TEST_CASE("corrected half connection coefficient differs from legacy one") {
  const double radius = 0.6;
  const teuk::KerrParameters parameters{1.0, 0.8, 1.4};
  const auto bg = teuk::kerr_background_point(parameters, radius, 0.0, 1.0);
  teuk::OrderedPairFields f{};
  teuk::OrderedPairDerivatives d{};
  f.F1 = Complex(0.7, -0.2);
  f.C2 = Complex(-0.3, 0.9);
  d.eth2_C2 = Complex(0.4, 0.1);

  const auto corrected =
      teuk::corrected_ordered_pair_source(radius, bg, f, d).D;
  const Complex connection =
      radius * radius *
      (Kokkos::conj(bg.pi0) + 2.0 * bg.tau0) * f.C2;
  const Complex correct_reference =
      f.F1 * (0.5 * radius * d.eth2_C2 + 0.5 * connection);
  const Complex legacy_reference =
      f.F1 * (0.5 * radius * d.eth2_C2 + connection);

  CHECK_COMPLEX_NEAR(corrected, correct_reference, 1.0e-14);
  CHECK(Kokkos::abs(corrected - legacy_reference) > 1.0e-4);
  CHECK_COMPLEX_NEAR(legacy_reference - corrected,
                     0.5 * f.F1 * connection, 1.0e-14);
}

TEST_CASE("Schwarzschild removes every disputed Kerr connection term") {
  const teuk::KerrParameters parameters{1.0, 0.0, 1.2};
  const auto bg = teuk::kerr_background_point(parameters, 0.5, 0.3, 0.7);
  CHECK_COMPLEX_NEAR(Kokkos::conj(bg.pi0) + 2.0 * bg.tau0,
                     Complex(0.0, 0.0), 1.0e-15);
}

TEST_CASE("outer source and coordinate forcing retain all four families") {
  const double radius = 0.36;
  const double cos_theta = 0.4;
  const double spin = 0.71;
  const double length = 1.3;
  const teuk::KerrBackgroundPoint bg{
      Complex(-0.8, 0.2), Complex(0.03, 0.07),
      Complex(-0.04, -0.06), Complex(-0.5, 0.1)};
  const teuk::InnerSource inner{Complex(0.4, -0.3), Complex(-0.2, 0.7)};
  const teuk::OuterSourceDerivatives derivatives{Complex(0.9, 0.1),
                                                  Complex(-0.6, 0.8)};
  const Complex expected_source =
      derivatives.delta3_D +
      radius * (4.0 * bg.mu0 + Kokkos::conj(bg.mu0)) * inner.D +
      radius * derivatives.ethprime3_T +
      radius * radius * (4.0 * bg.pi0 - Kokkos::conj(bg.tau0)) * inner.T;
  const Complex actual_source =
      teuk::outer_source_over_r3(radius, bg, inner, derivatives);
  CHECK_COMPLEX_NEAR(actual_source, expected_source, 1.0e-14);

  const double length4 = length * length * length * length;
  const Complex expected_forcing =
      2.0 * (length4 + spin * spin * radius * radius * cos_theta * cos_theta) *
      expected_source;
  CHECK_COMPLEX_NEAR(teuk::coordinate_second_order_forcing(
                         radius, cos_theta, spin, length, actual_source),
                     expected_forcing, 1.0e-14);
}

TEST_CASE("ordered-pair source executes in a Kokkos kernel") {
  Kokkos::View<Complex*> result("ordered pair result", 2);
  const teuk::KerrBackgroundPoint bg{Complex(-0.7, 0.1),
                                      Complex(0.02, 0.04),
                                      Complex(-0.01, -0.03),
                                      Complex(-0.5, 0.2)};
  teuk::OrderedPairFields f{};
  teuk::OrderedPairDerivatives d{};
  f.F1 = Complex(0.3, -0.2);
  f.C2 = Complex(-0.4, 0.8);
  d.eth2_C2 = Complex(0.7, 0.1);
  Kokkos::parallel_for(
      "ordered pair source", 1, KOKKOS_LAMBDA(const int) {
        const auto source =
            teuk::corrected_ordered_pair_source(0.4, bg, f, d);
        result(0) = source.D;
        result(1) = source.T;
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  const auto expected = teuk::corrected_ordered_pair_source(0.4, bg, f, d);
  CHECK_COMPLEX_NEAR(host(0), expected.D, 1.0e-14);
  CHECK_COMPLEX_NEAR(host(1), expected.T, 1.0e-14);
}

}  // namespace
