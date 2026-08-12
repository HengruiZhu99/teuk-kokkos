#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <limits>

#include "teuk/ghp.hpp"
#include "teuk/jet.hpp"
#include "teuk/plus2_bianchi.hpp"

namespace {

using C = teuk::Complex;
using J = teuk::Jet1<C>;

C scaled(const C& value, const double radius, const int power) {
  double factor = 1.0;
  for (int exponent = 0; exponent < power; ++exponent) factor *= radius;
  return factor * value;
}

TEST_CASE("plus2 Bianchi closures match ordinary-NP rescaling and signs") {
  const teuk::KerrParameters parameters{1.0, 0.91, 1.3};
  const double radius = 0.37;
  const double cos_theta = -0.29;
  const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
  const auto background = teuk::kerr_background_point(
      parameters, radius, cos_theta, sin_theta);
  const C z0(0.31, -0.22);
  const C z1(-0.18, 0.41);
  const C h(0.27, 0.09);
  const C sig(-0.13, 0.17);
  const C tau1(0.08, -0.04);
  const C c_sharp(-0.25, -0.12);
  const C b_sharp(0.16, 0.33);
  const C eth4_z1(0.43, -0.37);
  const C eth3_h(-0.21, 0.15);
  const C delta3_psi20(0.12, -0.28);
  const C ethprime3_psi20(-0.32, 0.19);

  const C f0 = teuk::plus2_bianchi_delta5_z0(
      radius, background, z0, z1, sig, eth4_z1);
  const C physical_f0 =
      scaled(eth4_z1, radius, 5) -
      scaled(background.mu0, radius, 1) * scaled(z0, radius, 5) -
      4.0 * scaled(background.tau0, radius, 2) *
          scaled(z1, radius, 4) +
      3.0 * scaled(sig, radius, 2) *
          scaled(background.psi20, radius, 3);
  CHECK_COMPLEX_NEAR(f0, physical_f0 / std::pow(radius, 5), 3.0e-14);

  // Regression against the former, nonlinear substitution
  // sigma^(1) Psi2^(1).  Bianchi-5 is a linear closure and therefore uses
  // the background type-D curvature Psi2^(0).
  const C wrong_perturbed_curvature =
      eth4_z1 - radius * background.mu0 * z0 -
      4.0 * radius * background.tau0 * z1 + 3.0 * sig * h;
  CHECK(Kokkos::abs(f0 - wrong_perturbed_curvature) > 1.0e-3);

  const C f1 = teuk::plus2_bianchi_delta4_z1(
      radius, background, z1, h, c_sharp, b_sharp, tau1, eth3_h,
      delta3_psi20, ethprime3_psi20);
  const C delta1_psi20 =
      -scaled(c_sharp, radius, 2) *
          scaled(delta3_psi20, radius, 3) +
      0.5 * scaled(b_sharp, radius, 1) *
          scaled(ethprime3_psi20, radius, 4);
  const C physical_f1 =
      scaled(eth3_h, radius, 4) -
      2.0 * scaled(background.mu0, radius, 1) *
          scaled(z1, radius, 4) -
      3.0 * scaled(background.tau0, radius, 2) * scaled(h, radius, 3) +
      delta1_psi20 -
      3.0 * scaled(tau1, radius, 2) *
          scaled(background.psi20, radius, 3);
  CHECK_COMPLEX_NEAR(f1, physical_f1 / std::pow(radius, 4), 3.0e-14);

  // Regression against the two erroneous metric-term signs.
  const C wrong_signs =
      eth3_h +
      radius * (-2.0 * background.mu0 * z1 -
                3.0 * background.tau0 * h +
                c_sharp * delta3_psi20 -
                0.5 * b_sharp * ethprime3_psi20 -
                3.0 * tau1 * background.psi20);
  CHECK(Kokkos::abs(f1 - wrong_signs) > 1.0e-3);
}

TEST_CASE("plus2 Bianchi closures have the exact Schwarzschild reduction") {
  const teuk::KerrParameters parameters{1.2, 0.0, 1.5};
  const double radius = 0.41;
  const auto background =
      teuk::kerr_background_point(parameters, radius, 0.2,
                                  std::sqrt(0.96));
  CHECK_COMPLEX_NEAR(background.tau0, C{}, 0.0);
  CHECK_COMPLEX_NEAR(background.pi0, C{}, 0.0);
  const C z0(0.1, 0.2), z1(-0.3, 0.4), h(0.5, -0.2);
  const C sig(0.07, 0.03), tau1(-0.04, 0.01);
  const C c_sharp(0.13, -0.08), b_sharp(-0.12, 0.05);
  const C eth4_z1(0.22, 0.11), eth3_h(-0.09, 0.14);
  const C delta3_psi20(0.17, -0.03), ethprime3_psi20{};
  CHECK_COMPLEX_NEAR(
      teuk::plus2_bianchi_delta5_z0(radius, background, z0, z1, sig,
                                    eth4_z1),
      eth4_z1 - radius * background.mu0 * z0 +
          3.0 * sig * background.psi20,
      2.0e-15);
  CHECK_COMPLEX_NEAR(
      teuk::plus2_bianchi_delta4_z1(
          radius, background, z1, h, c_sharp, b_sharp, tau1, eth3_h,
          delta3_psi20, ethprime3_psi20),
      eth3_h +
          radius * (-2.0 * background.mu0 * z1 -
                    c_sharp * delta3_psi20 -
                    3.0 * tau1 * background.psi20),
      2.0e-15);
}

TEST_CASE("plus2 Bianchi Jet tangents and triangular inversion are exact") {
  const teuk::KerrParameters parameters{1.0, -0.73, 1.25};
  const double radius = 0.36;
  const auto background = teuk::kerr_background_point(
      parameters, radius, 0.38, std::sqrt(1.0 - 0.38 * 0.38));
  const C z1(-0.21, 0.17), h(0.31, -0.11), cs(0.08, 0.19);
  const C bs(-0.14, 0.12), ta(0.06, -0.09), ethh(0.23, 0.04);
  const C z1r(0.16, -0.05), z1tr(-0.07, 0.13);
  const C ht(-0.04, 0.07), cst(0.02, -0.03), bst(-0.05, 0.01);
  const C tat(0.03, 0.02), ethht(-0.11, 0.06);
  const C dpsi(-0.18, 0.21), epsi(0.09, -0.15);

  const C f1 = teuk::plus2_bianchi_delta4_z1(
      radius, background, z1, h, cs, bs, ta, ethh, dpsi, epsi);
  const C z1t = teuk::plus2_invert_capital_delta_n(
      f1, z1, z1r, 4, radius, parameters.mass,
      parameters.compactification_length);
  const J f1_jet = teuk::plus2_bianchi_delta4_z1(
      radius, background, J{z1, z1t}, J{h, ht}, J{cs, cst}, J{bs, bst},
      J{ta, tat}, J{ethh, ethht}, dpsi, epsi);
  CHECK_COMPLEX_NEAR(f1_jet.value, f1, 2.0e-15);
  const C z1tt = teuk::plus2_invert_capital_delta_n(
      f1_jet.dt, z1t, z1tr, 4, radius, parameters.mass,
      parameters.compactification_length);
  CHECK_COMPLEX_NEAR(
      teuk::delta_n_point(z1, z1t, z1r, 4, radius, parameters.mass,
                          parameters.compactification_length),
      f1, 2.0e-15);
  CHECK_COMPLEX_NEAR(
      teuk::delta_n_point(z1t, z1tt, z1tr, 4, radius, parameters.mass,
                          parameters.compactification_length),
      f1_jet.dt, 2.0e-15);

  constexpr double epsilon = 1.0e-6;
  const auto shifted = [&](const double sign) {
    return teuk::plus2_bianchi_delta4_z1(
        radius, background, z1 + sign * epsilon * z1t,
        h + sign * epsilon * ht, cs + sign * epsilon * cst,
        bs + sign * epsilon * bst, ta + sign * epsilon * tat,
        ethh + sign * epsilon * ethht, dpsi, epsi);
  };
  CHECK_COMPLEX_NEAR(f1_jet.dt,
                     (shifted(1.0) - shifted(-1.0)) / (2.0 * epsilon),
                     8.0e-11);

  const C z0(0.19, -0.16), z0r(-0.12, 0.08), z0tr(0.05, -0.04);
  const C sig(-0.1, 0.13), sigt(0.04, 0.02);
  const C ethz1(0.2, -0.06), ethz1t(-0.03, 0.09);
  const C f0 = teuk::plus2_bianchi_delta5_z0(
      radius, background, z0, z1, sig, ethz1);
  const C z0t = teuk::plus2_invert_capital_delta_n(
      f0, z0, z0r, 5, radius, parameters.mass,
      parameters.compactification_length);
  const J f0_jet = teuk::plus2_bianchi_delta5_z0(
      radius, background, J{z0, z0t}, J{z1, z1t}, J{sig, sigt},
      J{ethz1, ethz1t});
  CHECK_COMPLEX_NEAR(
      f0_jet.dt,
      ethz1t - radius * background.mu0 * z0t -
          4.0 * radius * background.tau0 * z1t +
          3.0 * sigt * background.psi20,
      2.0e-15);
  const C z0tt = teuk::plus2_invert_capital_delta_n(
      f0_jet.dt, z0t, z0tr, 5, radius, parameters.mass,
      parameters.compactification_length);
  CHECK_COMPLEX_NEAR(
      teuk::delta_n_point(z0, z0t, z0r, 5, radius, parameters.mass,
                          parameters.compactification_length),
      f0, 2.0e-15);
  CHECK_COMPLEX_NEAR(
      teuk::delta_n_point(z0t, z0tt, z0tr, 5, radius, parameters.mass,
                          parameters.compactification_length),
      f0_jet.dt, 2.0e-15);
}

TEST_CASE("plus2 Bianchi inversion and peeling quotients are regular and fail closed") {
  const C f(0.7, -0.4), x(-0.2, 0.3), xr(0.1, 0.5);
  CHECK_COMPLEX_NEAR(
      teuk::plus2_invert_capital_delta_n(f, x, xr, 5, 0.0, 1.0, 1.4),
      0.5 * f, 0.0);
  const auto missing = teuk::plus2_peeling_quotient(C{}, 2, 0.0);
  CHECK(!missing.valid);
  const C coefficient(0.31, -0.27);
  const auto scri =
      teuk::plus2_peeling_quotient(C{}, 2, 0.0, true, coefficient);
  CHECK(scri.valid);
  CHECK_COMPLEX_NEAR(scri.value, coefficient, 0.0);
  const double radius = 0.23;
  const auto interior = teuk::plus2_peeling_quotient(
      radius * radius * coefficient, 2, radius);
  CHECK(interior.valid);
  CHECK_COMPLEX_NEAR(interior.value, coefficient, 2.0e-15);
  CHECK(!teuk::plus2_peeling_quotient(coefficient, 1, -0.1).valid);
  CHECK(!teuk::plus2_peeling_quotient(coefficient, 0, 0.2).valid);
  CHECK(!teuk::plus2_peeling_quotient(
             coefficient, 1, std::numeric_limits<double>::quiet_NaN())
             .valid);

  const teuk::KerrParameters parameters{1.0, 0.999, 1.1};
  const double horizon =
      parameters.compactification_length *
      parameters.compactification_length /
      (parameters.mass +
       std::sqrt(parameters.mass * parameters.mass -
                 parameters.spin * parameters.spin));
  const C horizon_value = teuk::plus2_invert_capital_delta_n(
      f, x, xr, 5, horizon, parameters.mass,
      parameters.compactification_length);
  CHECK(std::isfinite(horizon_value.real()));
  CHECK(std::isfinite(horizon_value.imag()));
}

TEST_CASE("plus2 Bianchi helpers have device parity on rotating Kerr") {
  Kokkos::View<C*[4]> output("plus2_bianchi_device", 1);
  const teuk::KerrParameters parameters{1.0, 0.999, 1.1};
  Kokkos::parallel_for(
      "plus2_bianchi_device_parity", 1, KOKKOS_LAMBDA(const int) {
        const double radius = 0.44;
        const double cosine = -0.33;
        const auto background = teuk::kerr_background_point(
            parameters, radius, cosine,
            Kokkos::sqrt(1.0 - cosine * cosine));
        output(0, 0) = teuk::plus2_bianchi_delta4_z1(
            radius, background, C(-0.2, 0.1), C(0.3, -0.4),
            C(0.1, 0.2), C(-0.3, 0.1), C(0.06, -0.02), C(0.2, 0.05),
            C(-0.1, 0.12), C(0.08, -0.09));
        output(0, 1) = teuk::plus2_bianchi_delta5_z0(
            radius, background, C(0.4, -0.1), C(-0.2, 0.1),
            C(0.07, 0.03), C(-0.05, 0.2));
        output(0, 2) = teuk::plus2_invert_capital_delta_n(
            output(0, 0), C(-0.2, 0.1), C(0.11, -0.07), 4, radius,
            parameters.mass, parameters.compactification_length);
        const auto quotient = teuk::plus2_peeling_quotient(
            C{}, 1, 0.0, true, C(0.17, -0.08));
        output(0, 3) = quotient.valid ? quotient.value : C(99.0, 0.0);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         output);
  const double radius = 0.44;
  const double cosine = -0.33;
  const auto background = teuk::kerr_background_point(
      parameters, radius, cosine, std::sqrt(1.0 - cosine * cosine));
  const C f1 = teuk::plus2_bianchi_delta4_z1(
      radius, background, C(-0.2, 0.1), C(0.3, -0.4), C(0.1, 0.2),
      C(-0.3, 0.1), C(0.06, -0.02), C(0.2, 0.05), C(-0.1, 0.12),
      C(0.08, -0.09));
  CHECK_COMPLEX_NEAR(host(0, 0), f1, 3.0e-14);
  CHECK_COMPLEX_NEAR(
      host(0, 1),
      teuk::plus2_bianchi_delta5_z0(
          radius, background, C(0.4, -0.1), C(-0.2, 0.1),
          C(0.07, 0.03), C(-0.05, 0.2)),
      3.0e-14);
  CHECK_COMPLEX_NEAR(
      host(0, 2),
      teuk::plus2_invert_capital_delta_n(
          f1, C(-0.2, 0.1), C(0.11, -0.07), 4, radius,
          parameters.mass, parameters.compactification_length),
      3.0e-14);
  CHECK_COMPLEX_NEAR(host(0, 3), C(0.17, -0.08), 2.0e-15);
}

}  // namespace
