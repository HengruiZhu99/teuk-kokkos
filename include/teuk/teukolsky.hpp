#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/grid.hpp"
#include "teuk/radial.hpp"
#include "teuk/types.hpp"

namespace teuk {

struct TeukolskyParameters {
  double mass = 1.0;
  double spin = 0.0;
  double compactification_length = 1.0;
  int spin_weight = -2;
  int azimuthal_mode = 0;
  double reduction_damping = 0.0;
};

struct TeukolskyCoefficients {
  double time = 0.0;              // C_T
  double radial_advection = 0.0;  // K
  double radial_principal = 0.0;  // H_R
  Complex definition = 0.0;      // G_m
  Complex q = 0.0;               // coefficient multiplying Q in P_t
  Complex psi = 0.0;             // coefficient multiplying psi in P_t
};

// Stationary point coefficients from section 7 of the corrected implementer
// reference.  Keeping this point function device-callable makes the exact same
// algebra usable in a future Kokkos multidimensional kernel.
KOKKOS_INLINE_FUNCTION TeukolskyCoefficients teukolsky_coefficients(
    const TeukolskyParameters& parameters, const double radius,
    const double theta) {
  const double mass = parameters.mass;
  const double spin = parameters.spin;
  const double length = parameters.compactification_length;
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const double radius2 = radius * radius;
  const double spin2 = spin * spin;
  const double mass_radius_over_length2 = mass * radius / length2;
  const Complex imaginary_unit(0.0, 1.0);

  TeukolskyCoefficients coefficients;
  coefficients.time =
      8.0 * mass * (2.0 * mass - spin2 * radius / length2) *
          (1.0 + 2.0 * mass_radius_over_length2) -
      spin2 * Kokkos::sin(theta) * Kokkos::sin(theta);
  coefficients.radial_advection =
      length2 - (8.0 * mass * mass - spin2) * radius2 / length2 +
      4.0 * spin2 * mass * radius * radius2 / length4;
  coefficients.radial_principal =
      radius2 / length4 *
      (length4 - 2.0 * length2 * mass * radius + spin2 * radius2);

  const double s = static_cast<double>(parameters.spin_weight);
  const double m = static_cast<double>(parameters.azimuthal_mode);
  coefficients.definition =
      2.0 * imaginary_unit * spin * m *
          (1.0 + 4.0 * mass_radius_over_length2) +
      2.0 *
          (2.0 * mass *
               (-s + (2.0 + s) * 2.0 * mass_radius_over_length2 -
                3.0 * spin2 * radius2 / length4) -
           spin2 * radius / length2 +
           imaginary_unit * s * spin * Kokkos::cos(theta));

  coefficients.q =
      2.0 * radius *
          (1.0 + s - (3.0 + s) * mass_radius_over_length2 +
           2.0 * spin2 * radius2 / length4) -
      2.0 * imaginary_unit * spin * m * radius2 / length2;
  coefficients.psi =
      -2.0 * radius *
          ((1.0 + s) * mass / length2 - spin2 * radius / length4) -
      2.0 * imaginary_unit * spin * m * radius / length2;
  return coefficients;
}

struct TeukolskyState {
  Complex P = 0.0;
  Complex Q = 0.0;
  Complex psi = 0.0;
};

KOKKOS_INLINE_FUNCTION Complex teukolsky_psi_rhs(
    const TeukolskyCoefficients& coefficients,
    const TeukolskyState& state) {
  return (state.P + 2.0 * coefficients.radial_advection * state.Q -
          coefficients.definition * state.psi) /
         coefficients.time;
}

KOKKOS_INLINE_FUNCTION Complex teukolsky_p_rhs(
    const TeukolskyCoefficients& coefficients,
    const TeukolskyState& state, const Complex& radial_derivative_q,
    const Complex& angular_laplacian_psi, const Complex& forcing) {
  return coefficients.radial_principal * radial_derivative_q +
         coefficients.q * state.Q + coefficients.psi * state.psi +
         angular_laplacian_psi + forcing;
}

KOKKOS_INLINE_FUNCTION Complex reduction_constraint(
    const TeukolskyState& state, const Complex& radial_derivative_psi) {
  return state.Q - radial_derivative_psi;
}

// Compact Q equation: differentiate the complete psi RHS with the selected
// radial operator, then damp the reduction constraint.  This deliberately
// avoids the error-prone expanded legacy Q coefficients.
KOKKOS_INLINE_FUNCTION Complex teukolsky_q_rhs(
    const Complex& radial_derivative_psi_rhs,
    const Complex& constraint, const double reduction_damping) {
  return radial_derivative_psi_rhs - reduction_damping * constraint;
}

enum class ReductionEvolution {
  FreeDamped,
  StageConstrained,
};

// Scratch for one (m,theta) radial line.  Construct it once and reuse it at
// every RK stage; evaluate_teukolsky_radial_line_rhs performs no allocation.
struct TeukolskyRadialWorkspace {
  explicit TeukolskyRadialWorkspace(const std::size_t point_count)
      : radial_derivative_psi(point_count),
        effective_q(point_count),
        radial_derivative_q(point_count),
        psi_velocity(point_count),
        radial_derivative_psi_velocity(point_count) {}

  [[nodiscard]] std::size_t size() const {
    return radial_derivative_psi.size();
  }

  std::vector<Complex> radial_derivative_psi;
  std::vector<Complex> effective_q;
  std::vector<Complex> radial_derivative_q;
  std::vector<Complex> psi_velocity;
  std::vector<Complex> radial_derivative_psi_velocity;
};

inline void enforce_reduction_constraint(
    const UniformRadialGrid& grid, std::vector<TeukolskyState>& state,
    TeukolskyRadialWorkspace& workspace) {
  if (state.size() != grid.size() || workspace.size() != grid.size()) {
    throw std::invalid_argument(
        "reduction projection buffers must match the radial grid");
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    workspace.psi_velocity[i] = state[i].psi;
  }
  fourth_order_radial_derivative(grid, workspace.psi_velocity,
                                 workspace.radial_derivative_psi);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    state[i].Q = workspace.radial_derivative_psi[i];
  }
}

// Evaluate a complete compact first-order reduction on one radial line.
// angular_laplacian_psi already contains {}_s Delta_Omega psi at the point;
// forcing is zero for first order and F^(2) for the driven system.
inline void evaluate_teukolsky_radial_line_rhs(
    const UniformRadialGrid& grid, const TeukolskyParameters& parameters,
    const double theta, const std::vector<TeukolskyState>& state,
    const std::vector<Complex>& angular_laplacian_psi,
    const std::vector<Complex>& forcing, const ReductionEvolution reduction,
    TeukolskyRadialWorkspace& workspace,
    std::vector<TeukolskyState>& rhs) {
  const std::size_t n = grid.size();
  if (state.size() != n || angular_laplacian_psi.size() != n ||
      forcing.size() != n || rhs.size() != n || workspace.size() != n) {
    throw std::invalid_argument(
        "Teukolsky radial-line buffers must match the radial grid");
  }
  if (!(parameters.compactification_length > 0.0)) {
    throw std::invalid_argument("compactification length must be positive");
  }

  for (std::size_t i = 0; i < n; ++i) {
    workspace.psi_velocity[i] = state[i].psi;
    workspace.effective_q[i] = state[i].Q;
  }
  fourth_order_radial_derivative(
      grid, workspace.psi_velocity, workspace.radial_derivative_psi);

  if (reduction == ReductionEvolution::StageConstrained) {
    for (std::size_t i = 0; i < n; ++i) {
      workspace.effective_q[i] = workspace.radial_derivative_psi[i];
    }
  }
  fourth_order_radial_derivative(grid, workspace.effective_q,
                                 workspace.radial_derivative_q);

  for (std::size_t i = 0; i < n; ++i) {
    const TeukolskyCoefficients coefficients = teukolsky_coefficients(
        parameters, grid.coordinate(i), theta);
    TeukolskyState effective_state = state[i];
    effective_state.Q = workspace.effective_q[i];
    workspace.psi_velocity[i] =
        teukolsky_psi_rhs(coefficients, effective_state);
  }
  fourth_order_radial_derivative(grid, workspace.psi_velocity,
                                 workspace.radial_derivative_psi_velocity);

  for (std::size_t i = 0; i < n; ++i) {
    const TeukolskyCoefficients coefficients = teukolsky_coefficients(
        parameters, grid.coordinate(i), theta);
    TeukolskyState effective_state = state[i];
    effective_state.Q = workspace.effective_q[i];
    rhs[i].psi = workspace.psi_velocity[i];
    rhs[i].P = teukolsky_p_rhs(
        coefficients, effective_state, workspace.radial_derivative_q[i],
        angular_laplacian_psi[i], forcing[i]);
    const Complex constraint =
        state[i].Q - workspace.radial_derivative_psi[i];
    const double damping =
        reduction == ReductionEvolution::FreeDamped
            ? parameters.reduction_damping
            : 0.0;
    rhs[i].Q = teukolsky_q_rhs(
        workspace.radial_derivative_psi_velocity[i], constraint, damping);
  }
}

inline double reduction_constraint_rms(
    const UniformRadialGrid& grid,
    const std::vector<TeukolskyState>& state,
    std::vector<Complex>& psi_buffer,
    std::vector<Complex>& derivative_buffer) {
  if (state.size() != grid.size() || psi_buffer.size() != grid.size() ||
      derivative_buffer.size() != grid.size()) {
    throw std::invalid_argument(
        "reduction diagnostic buffers must match the radial grid");
  }
  for (std::size_t i = 0; i < grid.size(); ++i) psi_buffer[i] = state[i].psi;
  fourth_order_radial_derivative(grid, psi_buffer, derivative_buffer);
  double norm2 = 0.0;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const Complex residual = state[i].Q - derivative_buffer[i];
    norm2 += residual.real() * residual.real() +
             residual.imag() * residual.imag();
  }
  return Kokkos::sqrt(norm2 / static_cast<double>(grid.size()));
}

}  // namespace teuk
