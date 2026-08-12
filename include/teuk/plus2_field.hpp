#pragma once

#include <Kokkos_Core.hpp>

#include "teuk/teukolsky.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Ripley et al. arXiv:2010.00162 Eq. (21b), on the continuous branch
// (Psi2/M)^(1/3)=-(r-i a cos(theta))^(-1), in compact radius R=L^2/r:
//
//   Psi0_code = W_plus Z_plus,
//   W_plus = R^5/(L^2-i a R cos(theta))^4.
//
// The code tetrad is the horizon-regular tetrad of that paper.  This helper
// deliberately takes cos(theta), matching the coordinate arrays already kept
// by the spatial pipeline and avoiding a second trigonometric evaluation.
KOKKOS_INLINE_FUNCTION Complex plus2_code_tetrad_scaling(
    const double radius, const double cos_theta, const double spin,
    const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  const double radius2 = radius * radius;
  const double radius4 = radius2 * radius2;
  const Complex denominator(length2, -spin * radius * cos_theta);
  const Complex denominator2 = denominator * denominator;
  return (radius4 * radius) / (denominator2 * denominator2);
}

KOKKOS_INLINE_FUNCTION Complex plus2_code_tetrad_scaling(
    const TeukolskyParameters& parameters, const double radius,
    const double cos_theta) {
  return plus2_code_tetrad_scaling(
      radius, cos_theta, parameters.spin,
      parameters.compactification_length);
}

KOKKOS_INLINE_FUNCTION Complex plus2_code_tetrad_psi0(
    const Complex& regularized_plus2, const TeukolskyParameters& parameters,
    const double radius, const double cos_theta) {
  return plus2_code_tetrad_scaling(parameters, radius, cos_theta) *
         regularized_plus2;
}

// Interior/horizon inverse.  At scri W_plus=0 and Z_plus is instead the
// finite peeling coefficient; callers must obtain that value by an analytic
// limit, not by forming the indeterminate ratio Psi0/W_plus at R=0.
KOKKOS_INLINE_FUNCTION Complex plus2_regularized_from_code_tetrad_interior(
    const Complex& code_tetrad_psi0,
    const TeukolskyParameters& parameters, const double radius,
    const double cos_theta) {
  return code_tetrad_psi0 /
         plus2_code_tetrad_scaling(parameters, radius, cos_theta);
}

KOKKOS_INLINE_FUNCTION double plus2_scri_scaling_coefficient(
    const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  const double length4 = length2 * length2;
  return 1.0 / (length4 * length4);
}

}  // namespace teuk
