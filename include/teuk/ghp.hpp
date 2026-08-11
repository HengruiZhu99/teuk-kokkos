#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace teuk {

using Complex = Kokkos::complex<double>;

// Delta_n X = R^{-n} Delta(R^n X), evaluated from point values and first
// coordinate derivatives. This is valid for any spin and boost weights.
KOKKOS_INLINE_FUNCTION
Complex delta_n_point(const Complex& value, const Complex& dt_value,
                      const Complex& dr_value, const int radial_falloff,
                      const double radius, const double mass,
                      const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  const double time_coefficient = 2.0 + 4.0 * mass * radius / length2;
  const double radial_coefficient = radius / length2;
  return time_coefficient * dt_value +
         radial_coefficient *
             (radius * dr_value + radial_falloff * value);
}

// eth_n at one point. raised_value is R_s X in the same modal/collocation
// representation as value. The radial-falloff label n does not appear in the
// point formula because the angular tetrad vector has no radial derivative.
KOKKOS_INLINE_FUNCTION
Complex eth_n_point(const Complex& value, const Complex& dt_value,
                    const Complex& raised_value, const int spin_weight,
                    const int boost_weight, const double radius,
                    const double sin_theta, const double cos_theta,
                    const double kerr_spin,
                    const double compactification_length) {
  const Complex imaginary_unit(0.0, 1.0);
  const double length2 =
      compactification_length * compactification_length;
  const double sqrt_two = Kokkos::sqrt(2.0);
  const int p_weight = spin_weight + boost_weight;
  const Complex denominator(length2, -kerr_spin * radius * cos_theta);

  const Complex derivative_term =
      (-imaginary_unit * kerr_spin * sin_theta * dt_value + raised_value) /
      (sqrt_two * denominator);
  const Complex connection_term =
      -imaginary_unit * static_cast<double>(p_weight) * kerr_spin * radius *
      sin_theta * value / (sqrt_two * denominator * denominator);
  return derivative_term + connection_term;
}

// eth'_n at one point. lowered_value is L_s X, including the conventional
// minus sign in L_s {}_sY_lm.
KOKKOS_INLINE_FUNCTION
Complex ethprime_n_point(const Complex& value, const Complex& dt_value,
                         const Complex& lowered_value,
                         const int spin_weight, const int boost_weight,
                         const double radius, const double sin_theta,
                         const double cos_theta, const double kerr_spin,
                         const double compactification_length) {
  const Complex imaginary_unit(0.0, 1.0);
  const double length2 =
      compactification_length * compactification_length;
  const double sqrt_two = Kokkos::sqrt(2.0);
  const int q_weight = -spin_weight + boost_weight;
  const Complex denominator(length2, kerr_spin * radius * cos_theta);

  const Complex derivative_term =
      (imaginary_unit * kerr_spin * sin_theta * dt_value + lowered_value) /
      (sqrt_two * denominator);
  const Complex connection_term =
      imaginary_unit * static_cast<double>(q_weight) * kerr_spin * radius *
      sin_theta * value / (sqrt_two * denominator * denominator);
  return derivative_term + connection_term;
}

}  // namespace teuk
