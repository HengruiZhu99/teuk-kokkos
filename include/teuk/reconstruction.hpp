#pragma once

#include <Kokkos_Complex.hpp>

#include "teuk/background.hpp"

namespace teuk {

// Point values of the seven reconstruction fields. Sharp quantities must be
// supplied from conjugate(-m), never from conjugate(+m) implicitly.
struct ReconstructionFields {
  Complex F;
  Complex G;
  Complex H;
  Complex Lambda;
  Complex Pi;
  Complex B;
  Complex C;
  Complex U;
  Complex Pi_sharp;
  Complex B_sharp;
  Complex C_sharp;
};

struct ReconstructionAngularDerivatives {
  Complex eth1_F;
  Complex eth2_G;
  Complex eth2_C;
  Complex eth2_Pi;
  Complex ethprime1_B_sharp;
  Complex ethprime2_C_sharp;
};

// Right sides R_X in Delta_n X = R_X, in reconstruction order
// (G, Lambda, H, B, Pi, C, U).
struct ReconstructionDeltaRhs {
  Complex G;
  Complex Lambda;
  Complex H;
  Complex B;
  Complex Pi;
  Complex C;
  Complex U;
};

KOKKOS_INLINE_FUNCTION
ReconstructionDeltaRhs reconstruction_delta_rhs(
    const double radius, const KerrBackgroundPoint& background,
    const ReconstructionFields& fields,
    const ReconstructionAngularDerivatives& angular) {
  const Complex mu0_bar = Kokkos::conj(background.mu0);
  const Complex pi0_bar = Kokkos::conj(background.pi0);
  const Complex tau0_bar = Kokkos::conj(background.tau0);
  const double radius2 = radius * radius;

  ReconstructionDeltaRhs rhs;
  rhs.G = -4.0 * radius * background.mu0 * fields.G + angular.eth1_F -
          radius * background.tau0 * fields.F;

  rhs.Lambda = -radius * (background.mu0 + mu0_bar) * fields.Lambda -
               fields.F;

  rhs.H = -3.0 * radius * background.mu0 * fields.H + angular.eth2_G -
          2.0 * radius * background.tau0 * fields.G;

  rhs.B = radius * (background.mu0 - mu0_bar) * fields.B -
          2.0 * fields.Lambda;

  rhs.Pi = -fields.G -
           radius * (pi0_bar + background.tau0) * fields.Lambda +
           0.5 * radius2 * background.mu0 *
               (pi0_bar + background.tau0) * fields.B;

  rhs.C = -radius * mu0_bar * fields.C - 2.0 * fields.Pi -
          radius * background.tau0 * fields.B;

  // The U equation is deliberately written one source family per term.
  const Complex u_transport = -radius * mu0_bar * fields.U;
  const Complex c_eth =
      -radius * background.mu0 * angular.eth2_C;
  const Complex c_connection =
      -radius2 * background.mu0 * (pi0_bar + 2.0 * background.tau0) *
      fields.C;
  const Complex pi_eth = -2.0 * angular.eth2_Pi;
  const Complex pi_connection = -2.0 * radius * pi0_bar * fields.Pi;
  const Complex psi2_term = -2.0 * fields.H;
  const Complex sharp_pi_term =
      -2.0 * radius * background.pi0 * fields.Pi_sharp;
  const Complex sharp_b_eth =
      -radius * background.pi0 * angular.ethprime1_B_sharp;
  const Complex sharp_b_connection =
      radius2 * background.pi0 * background.pi0 * fields.B_sharp;
  const Complex sharp_c_eth =
      radius * background.mu0 * angular.ethprime2_C_sharp;
  const Complex sharp_c_connection =
      radius2 * (-3.0 * background.mu0 * background.pi0 +
                 2.0 * mu0_bar * background.pi0 -
                 2.0 * background.mu0 * tau0_bar) *
      fields.C_sharp;

  rhs.U = u_transport + c_eth + c_connection + pi_eth + pi_connection +
          psi2_term + sharp_pi_term + sharp_b_eth + sharp_b_connection +
          sharp_c_eth + sharp_c_connection;
  return rhs;
}

// Solve Delta_n X=R_X for partial_T X after a radial derivative has been
// supplied by the chosen finite-difference/SBP operator.
KOKKOS_INLINE_FUNCTION
Complex reconstruction_time_derivative(
    const Complex& value, const Complex& radial_derivative,
    const Complex& delta_rhs, const int radial_falloff, const double radius,
    const double mass, const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  const double denominator = 2.0 + 4.0 * mass * radius / length2;
  const Complex radial_terms =
      (radius * radius / length2) * radial_derivative +
      (static_cast<double>(radial_falloff) * radius / length2) * value;
  return (delta_rhs - radial_terms) / denominator;
}

}  // namespace teuk
