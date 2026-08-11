#pragma once

#include <Kokkos_Complex.hpp>

#include "teuk/background.hpp"
#include "teuk/types.hpp"

namespace teuk {

// These are the three independent reconstruction checks monitored by the
// legacy formulation. Unlike the seven transport consistency residuals, none
// is an equation used to define the reconstruction time derivative.
struct IndependentReconstructionConstraints {
  Complex psi3_bianchi;
  Complex psi2_bianchi;
  Complex hll_reality;
};

KOKKOS_INLINE_FUNCTION
IndependentReconstructionConstraints independent_reconstruction_constraints_point(
    const double radius, const KerrBackgroundPoint& background,
    const Complex& F, const Complex& G, const Complex& H,
    const Complex& Lambda, const Complex& Pi, const Complex& B,
    const Complex& C, const Complex& U, const Complex& U_sharp,
    const Complex& thorn1_F, const Complex& thorn2_G,
    const Complex& ethprime2_G, const Complex& ethprime3_H) {
  const double radius2 = radius * radius;
  const double radius3 = radius2 * radius;
  IndependentReconstructionConstraints residuals;
  residuals.psi3_bianchi =
      radius * ethprime2_G + 4.0 * radius2 * background.pi0 * G -
      thorn1_F + background.rho0 * F -
      3.0 * radius2 * background.psi20 * Lambda;
  residuals.psi2_bianchi =
      background.psi20 *
          (-3.0 * radius3 * background.mu0 * C -
           1.5 * radius3 * background.tau0 * B - 3.0 * radius2 * Pi) -
      radius * ethprime3_H - 3.0 * radius2 * background.pi0 * H +
      thorn2_G - 2.0 * background.rho0 * G;
  residuals.hll_reality =
      U / background.mu0 - U_sharp / Kokkos::conj(background.mu0);
  return residuals;
}

}  // namespace teuk
