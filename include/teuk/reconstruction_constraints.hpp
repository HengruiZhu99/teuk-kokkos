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

struct ReconstructionConstraintEquationTerms {
  Complex left;
  Complex right;
};

struct IndependentReconstructionConstraintTerms {
  ReconstructionConstraintEquationTerms psi3_bianchi;
  ReconstructionConstraintEquationTerms psi2_bianchi;
  ReconstructionConstraintEquationTerms hll_reality;
};

KOKKOS_INLINE_FUNCTION
IndependentReconstructionConstraintTerms
independent_reconstruction_constraint_terms_point(
    const double radius, const KerrBackgroundPoint& background,
    const Complex& F, const Complex& G, const Complex& H,
    const Complex& Lambda, const Complex& Pi, const Complex& B,
    const Complex& C, const Complex& U, const Complex& U_sharp,
    const Complex& thorn1_F, const Complex& thorn2_G,
    const Complex& ethprime2_G, const Complex& ethprime3_H) {
  const double radius2 = radius * radius;
  const double radius3 = radius2 * radius;
  IndependentReconstructionConstraintTerms terms;
  terms.psi3_bianchi.left =
      radius * ethprime2_G + 4.0 * radius2 * background.pi0 * G +
      background.rho0 * F;
  terms.psi3_bianchi.right =
      thorn1_F + 3.0 * radius2 * background.psi20 * Lambda;
  const Complex psi2_background_terms =
      background.psi20 *
      (-3.0 * radius3 * background.mu0 * C -
       1.5 * radius3 * background.tau0 * B - 3.0 * radius2 * Pi);
  terms.psi2_bianchi.left = thorn2_G - 2.0 * background.rho0 * G;
  terms.psi2_bianchi.right =
      radius * ethprime3_H + 3.0 * radius2 * background.pi0 * H -
      psi2_background_terms;
  terms.hll_reality.left = U / background.mu0;
  terms.hll_reality.right = U_sharp / Kokkos::conj(background.mu0);
  return terms;
}

KOKKOS_INLINE_FUNCTION
IndependentReconstructionConstraints independent_reconstruction_constraints_point(
    const double radius, const KerrBackgroundPoint& background,
    const Complex& F, const Complex& G, const Complex& H,
    const Complex& Lambda, const Complex& Pi, const Complex& B,
    const Complex& C, const Complex& U, const Complex& U_sharp,
    const Complex& thorn1_F, const Complex& thorn2_G,
    const Complex& ethprime2_G, const Complex& ethprime3_H) {
  const auto terms = independent_reconstruction_constraint_terms_point(
      radius, background, F, G, H, Lambda, Pi, B, C, U, U_sharp,
      thorn1_F, thorn2_G, ethprime2_G, ethprime3_H);
  IndependentReconstructionConstraints residuals;
  residuals.psi3_bianchi =
      terms.psi3_bianchi.left - terms.psi3_bianchi.right;
  residuals.psi2_bianchi =
      terms.psi2_bianchi.left - terms.psi2_bianchi.right;
  residuals.hll_reality =
      terms.hll_reality.left - terms.hll_reality.right;
  return residuals;
}

}  // namespace teuk
