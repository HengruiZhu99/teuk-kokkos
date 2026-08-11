#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace teuk {

using Complex = Kokkos::complex<double>;

struct KerrParameters {
  double mass = 1.0;
  double spin = 0.0;
  double compactification_length = 1.0;
};

// Rescaled rotated-Kinnersley background coefficients at one (R, theta) point.
// The physical coefficients are mu=R*mu0, tau=R^2*tau0,
// pi=R^2*pi0, and Psi2=R^3*psi20.
struct KerrBackgroundPoint {
  Complex mu0 = 0.0;
  Complex tau0 = 0.0;
  Complex pi0 = 0.0;
  Complex psi20 = 0.0;
  Complex rho0 = 0.0;
  Complex epsilon0 = 0.0;
};

KOKKOS_INLINE_FUNCTION
KerrBackgroundPoint kerr_background_point(const KerrParameters& parameters,
                                           const double radius,
                                           const double cos_theta,
                                           const double sin_theta) {
  const double mass = parameters.mass;
  const double spin = parameters.spin;
  const double length = parameters.compactification_length;
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const Complex imaginary_unit(0.0, 1.0);
  const double sqrt_two = Kokkos::sqrt(2.0);

  const Complex minus_radial_denominator(-length2,
                                          spin * radius * cos_theta);
  const Complex angular_denominator(length2,
                                     -spin * radius * cos_theta);
  const Complex angular_denominator_squared =
      angular_denominator * angular_denominator;
  const double real_pi_denominator =
      length4 + spin * spin * radius * radius * cos_theta * cos_theta;

  KerrBackgroundPoint background;
  background.mu0 = Complex(1.0, 0.0) / minus_radial_denominator;
  background.tau0 = imaginary_unit * spin * sin_theta /
                    (sqrt_two * angular_denominator_squared);
  background.pi0 = -imaginary_unit * spin * sin_theta /
                   (sqrt_two * real_pi_denominator);
  const Complex ingoing_denominator(length2,
                                     spin * radius * cos_theta);
  const Complex optical_denominator =
      angular_denominator_squared * ingoing_denominator;
  background.rho0 =
      -0.5 * (length4 - 2.0 * length2 * mass * radius +
              spin * spin * radius * radius) /
      optical_denominator;
  background.epsilon0 =
      0.5 *
      Complex(length2 * mass - spin * spin * radius,
              -spin * (length2 - mass * radius) * cos_theta) /
      optical_denominator;
  background.psi20 = -mass /
                     (angular_denominator_squared * angular_denominator);
  return background;
}

KOKKOS_INLINE_FUNCTION
KerrBackgroundPoint kerr_background_point(const KerrParameters& parameters,
                                           const double radius,
                                           const double theta) {
  return kerr_background_point(parameters, radius, Kokkos::cos(theta),
                               Kokkos::sin(theta));
}

// Exact type-D identity used to evolve mu*h_ll instead of h_ll.
KOKKOS_INLINE_FUNCTION
Complex delta_mu_from_identity(const Complex& physical_mu) {
  return -physical_mu * physical_mu;
}

}  // namespace teuk
