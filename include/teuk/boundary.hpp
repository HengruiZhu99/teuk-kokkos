#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "teuk/teukolsky.hpp"

namespace teuk {

// Eigenvalues of A in the convention partial_T U = A partial_R U + lower
// order terms. Coordinate propagation velocities are -lambda. The complete
// (P,Q,psi) principal matrix is block triangular, so the complex coefficient
// multiplying partial_R psi changes eigenvectors but not these eigenvalues.
struct RadialPrincipalCharacteristics {
  double lambda_minus = 0.0;
  double lambda_zero = 0.0;
  double lambda_plus = 0.0;
  double coordinate_velocity_minus = 0.0;
  double coordinate_velocity_zero = 0.0;
  double coordinate_velocity_plus = 0.0;
};

KOKKOS_INLINE_FUNCTION RadialPrincipalCharacteristics
radial_principal_characteristics(
    const TeukolskyCoefficients& coefficients) {
  const double discriminant =
      coefficients.radial_advection * coefficients.radial_advection +
      coefficients.time * coefficients.radial_principal;
  const double root = Kokkos::sqrt(discriminant);
  const double lambda_minus =
      (coefficients.radial_advection - root) / coefficients.time;
  const double lambda_plus =
      (coefficients.radial_advection + root) / coefficients.time;
  return {lambda_minus, 0.0, lambda_plus, -lambda_minus, 0.0,
          -lambda_plus};
}

KOKKOS_INLINE_FUNCTION RadialPrincipalCharacteristics
radial_principal_characteristics(
    const TeukolskyParameters& parameters, const double radius,
    const double theta) {
  return radial_principal_characteristics(
      teukolsky_coefficients(parameters, radius, theta));
}

enum class RadialBoundarySide { ScriLower, HorizonUpper };

struct RadialBoundaryClassification {
  int incoming = 0;
  int outgoing = 0;
  int stationary = 0;
};

KOKKOS_INLINE_FUNCTION RadialBoundaryClassification
classify_radial_boundary(const RadialPrincipalCharacteristics& characteristics,
                         const RadialBoundarySide side,
                         const double tolerance = 1.0e-13) {
  const double outward_normal =
      side == RadialBoundarySide::ScriLower ? -1.0 : 1.0;
  const double velocities[3] = {characteristics.coordinate_velocity_minus,
                                characteristics.coordinate_velocity_zero,
                                characteristics.coordinate_velocity_plus};
  RadialBoundaryClassification result;
  for (const double velocity : velocities) {
    const double outward_velocity = outward_normal * velocity;
    if (outward_velocity > tolerance) {
      ++result.outgoing;
    } else if (outward_velocity < -tolerance) {
      ++result.incoming;
    } else {
      ++result.stationary;
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION double compactified_outer_horizon_radius(
    const TeukolskyParameters& parameters) {
  const double discriminant =
      parameters.mass * parameters.mass - parameters.spin * parameters.spin;
  const double r_plus =
      parameters.mass + Kokkos::sqrt(discriminant);
  return parameters.compactification_length *
         parameters.compactification_length / r_plus;
}

// The natural symmetrizer for the propagating (P,Q) block is
// diag(1,C_T H_R): S A is symmetric with off-diagonal entry H_R. It is
// positive in the open exterior but degenerates at both endpoints because
// H_R=0. These weights are exposed to make that blocker directly testable.
struct PropagatingSymmetrizerWeights {
  double P = 1.0;
  double Q = 0.0;
};

KOKKOS_INLINE_FUNCTION PropagatingSymmetrizerWeights
propagating_symmetrizer_weights(
    const TeukolskyCoefficients& coefficients) {
  return {1.0, coefficients.time * coefficients.radial_principal};
}

// Without added dissipation, the compact reduction gives exactly
// partial_T C_Q = -gamma_Q C_Q for C_Q=Q-partial_R psi. Independent
// dissipation on Q and psi adds Dcal(Q)-D_R(Dcal(psi)); see sbp.hpp and the
// boundary documentation. The continuum constraint is stationary rather than
// an incoming transport field at either radial endpoint.
KOKKOS_INLINE_FUNCTION Complex reduction_constraint_time_derivative(
    const Complex& constraint, const double reduction_damping) {
  return -reduction_damping * constraint;
}

}  // namespace teuk
