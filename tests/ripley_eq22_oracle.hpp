#pragma once

#include <cmath>
#include <complex>

namespace teuk::test::ripley_eq22 {

// Independent host transcription of Ripley et al. arXiv:2010.00162
// Eqs. (22)--(23), specialized to s=+2 and exp(i m phi).  This oracle uses
// neither teukolsky_coefficients nor Kokkos arithmetic.
struct Parameters {
  double mass;
  double spin;
  double compactification_length;
  int azimuthal_mode;
};

struct FirstOrderCoefficients {
  double time;
  double radial_advection;
  double radial_principal;
  std::complex<double> definition;
  std::complex<double> q;
  std::complex<double> psi;
};

inline FirstOrderCoefficients coefficients(const Parameters& parameters,
                                           const double radius,
                                           const double theta) {
  constexpr double spin_weight = 2.0;
  const double mass = parameters.mass;
  const double spin = parameters.spin;
  const double length = parameters.compactification_length;
  const double m = static_cast<double>(parameters.azimuthal_mode);
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const double radius2 = radius * radius;
  const double spin2 = spin * spin;
  const std::complex<double> imaginary_unit(0.0, 1.0);

  // Eq. (22), line 1: coefficients of d_T^2, -2 d_T d_R,
  // and -d_R^2.  The first-order reduction of Eq. (23) uses +H_R d_R Q.
  const double c_t =
      8.0 * mass * (2.0 * mass - spin2 * radius / length2) *
          (1.0 + 2.0 * mass * radius / length2) -
      spin2 * std::sin(theta) * std::sin(theta);
  const double k =
      length2 - (8.0 * mass * mass - spin2) * radius2 / length2 +
      4.0 * spin2 * mass * radius * radius2 / length4;
  const double h_r =
      radius2 / length4 *
      (length4 - 2.0 * length2 * mass * radius + spin2 * radius2);

  // Eq. (23a): coefficient grouped with psi in the definition of P.
  const std::complex<double> g_m =
      2.0 * imaginary_unit * spin * m *
          (1.0 + 4.0 * mass * radius / length2) +
      2.0 *
          (2.0 * mass *
               (-spin_weight + (2.0 + spin_weight) * 2.0 * mass * radius /
                                   length2 -
                3.0 * spin2 * radius2 / length4) -
           spin2 * radius / length2 +
           imaginary_unit * spin_weight * spin * std::cos(theta));

  // The last line of Eq. (22), moved to the right-hand side of d_T P.
  const std::complex<double> q =
      2.0 * radius *
          (1.0 + spin_weight -
           (3.0 + spin_weight) * mass * radius / length2 +
           2.0 * spin2 * radius2 / length4) -
      2.0 * imaginary_unit * spin * m * radius2 / length2;
  const std::complex<double> psi =
      -2.0 * radius *
          ((1.0 + spin_weight) * mass / length2 -
           spin2 * radius / length4) -
      2.0 * imaginary_unit * spin * m * radius / length2;

  return {c_t, k, h_r, g_m, q, psi};
}

}  // namespace teuk::test::ripley_eq22
