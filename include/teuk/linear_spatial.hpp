#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/fields.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/sbp.hpp"
#include "teuk/teukolsky.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class TeukolskyRadialScratch : std::size_t {
  RadialDerivativePsi = 0,
  EffectiveQ = 1,
  RadialDerivativeQ = 2,
  PsiVelocity = 3,
  RadialDerivativePsiVelocity = 4,
  Count = 5,
};

using TeukolskyRadialStateView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using TeukolskyRadialValueView =
    Kokkos::View<Complex**, Kokkos::LayoutRight, MemorySpace>;
using TeukolskyRadialScratchView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using SignedModeView = Kokkos::View<int*, MemorySpace>;

// Device-resident storage for a fixed-theta collection of signed-m radial
// lines.  Logical ordering is (mode,field,radial), so every radial line is
// contiguous and can be passed to the allocation-free SBP point kernels.
// Angular Laplacians and forcing are stage inputs populated by the caller.
class TeukolskyRadialLines {
 public:
  TeukolskyRadialLines(const ModeRegistry& registry,
                       const UniformRadialGrid& grid,
                       const std::string& label = "teukolsky_radial_lines")
      : grid_(grid),
        modes_(label + "_modes", registry.size()),
        state_(label + "_state", registry.size(),
               static_cast<std::size_t>(TeukolskyField::Count), grid.size()),
        rhs_(label + "_rhs", registry.size(),
             static_cast<std::size_t>(TeukolskyField::Count), grid.size()),
        angular_laplacian_(label + "_angular_laplacian", registry.size(),
                           grid.size()),
        forcing_(label + "_forcing", registry.size(), grid.size()),
        scratch_(label + "_scratch", registry.size(),
                 static_cast<std::size_t>(TeukolskyRadialScratch::Count),
                 grid.size()) {
    if (grid.size() < d42_minimum_points) {
      throw std::invalid_argument(
          "SBP Teukolsky radial lines require at least eight points");
    }
    auto host_modes = Kokkos::create_mirror_view(modes_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_modes(mode) = registry.modes()[mode];
    }
    Kokkos::deep_copy(modes_, host_modes);
  }

  [[nodiscard]] std::size_t mode_count() const { return modes_.extent(0); }
  [[nodiscard]] std::size_t radial_point_count() const {
    return grid_.size();
  }
  [[nodiscard]] const UniformRadialGrid& grid() const { return grid_; }

  [[nodiscard]] SignedModeView modes() const { return modes_; }
  [[nodiscard]] TeukolskyRadialStateView state() const { return state_; }
  [[nodiscard]] TeukolskyRadialStateView rhs() const { return rhs_; }
  [[nodiscard]] TeukolskyRadialValueView angular_laplacian() const {
    return angular_laplacian_;
  }
  [[nodiscard]] TeukolskyRadialValueView forcing() const { return forcing_; }
  [[nodiscard]] TeukolskyRadialScratchView scratch() const { return scratch_; }

 private:
  UniformRadialGrid grid_;
  SignedModeView modes_;
  TeukolskyRadialStateView state_;
  TeukolskyRadialStateView rhs_;
  TeukolskyRadialValueView angular_laplacian_;
  TeukolskyRadialValueView forcing_;
  TeukolskyRadialScratchView scratch_;
};

// Core D4-2 method-of-lines RHS over all signed modes at one theta point. The
// caller owns every View and the execution-space instance, so RK stage Views
// and a queue shared by a larger pipeline can be supplied directly. The three
// launches preserve their data dependencies without host copies or timestep
// allocations.
// No physical SAT penalty is applied. The endpoint audit in boundary.hpp finds
// one outgoing and two stationary modes at each end, with no incoming mode;
// its natural continuum symmetrizer is endpoint-degenerate, so a nonzero SAT
// remains blocked pending a full semi-discrete energy/normal-mode analysis.
template <class SpatialExecutionSpace, class ModeView, class StateView,
          class OutputView, class ScratchView, class AngularView,
          class ForcingView>
void evaluate_sbp_teukolsky_radial_lines_rhs(
    const SpatialExecutionSpace& execution, const UniformRadialGrid& grid,
    const ModeView modes, const StateView state, const OutputView rhs,
    const ScratchView scratch, const AngularView angular_laplacian,
    const ForcingView forcing,
    const TeukolskyParameters& base_parameters, const double theta,
    const ReductionEvolution reduction,
    const double dissipation_strength = 0.0) {
  static_assert(
      std::is_same_v<typename StateView::array_layout, Kokkos::LayoutRight>,
      "SBP stage state must have contiguous radial LayoutRight storage");
  static_assert(
      std::is_same_v<typename OutputView::array_layout, Kokkos::LayoutRight>,
      "SBP stage output must have contiguous radial LayoutRight storage");
  static_assert(
      std::is_same_v<typename ScratchView::array_layout, Kokkos::LayoutRight>,
      "SBP scratch must have contiguous radial LayoutRight storage");
  if (!(base_parameters.compactification_length > 0.0)) {
    throw std::invalid_argument("compactification length must be positive");
  }
  if (dissipation_strength < 0.0) {
    throw std::invalid_argument("dissipation strength must be nonnegative");
  }

  const std::size_t point_count = grid.size();
  const std::size_t mode_count = modes.extent(0);
  if (point_count < d42_minimum_points || state.extent(0) != mode_count ||
      rhs.extent(0) != mode_count || scratch.extent(0) != mode_count ||
      angular_laplacian.extent(0) != mode_count ||
      forcing.extent(0) != mode_count ||
      state.extent(1) != static_cast<std::size_t>(TeukolskyField::Count) ||
      rhs.extent(1) != static_cast<std::size_t>(TeukolskyField::Count) ||
      scratch.extent(1) !=
          static_cast<std::size_t>(TeukolskyRadialScratch::Count) ||
      state.extent(2) != point_count || rhs.extent(2) != point_count ||
      scratch.extent(2) != point_count ||
      angular_laplacian.extent(1) != point_count ||
      forcing.extent(1) != point_count) {
    throw std::invalid_argument(
        "SBP Teukolsky stage Views do not match modes, fields, and grid");
  }
  const std::size_t total_points = mode_count * point_count;
  const double inverse_spacing = 1.0 / grid.spacing();
  constexpr std::size_t p_field =
      static_cast<std::size_t>(TeukolskyField::P);
  constexpr std::size_t q_field =
      static_cast<std::size_t>(TeukolskyField::Q);
  constexpr std::size_t psi_field =
      static_cast<std::size_t>(TeukolskyField::Psi);
  constexpr std::size_t dr_psi =
      static_cast<std::size_t>(TeukolskyRadialScratch::RadialDerivativePsi);
  constexpr std::size_t effective_q =
      static_cast<std::size_t>(TeukolskyRadialScratch::EffectiveQ);
  constexpr std::size_t dr_q =
      static_cast<std::size_t>(TeukolskyRadialScratch::RadialDerivativeQ);
  constexpr std::size_t psi_velocity =
      static_cast<std::size_t>(TeukolskyRadialScratch::PsiVelocity);
  constexpr std::size_t dr_psi_velocity = static_cast<std::size_t>(
      TeukolskyRadialScratch::RadialDerivativePsiVelocity);

  const Kokkos::RangePolicy<SpatialExecutionSpace> policy(
      execution, 0,
      static_cast<typename SpatialExecutionSpace::size_type>(total_points));
  Kokkos::parallel_for(
      "teuk_sbp_prepare_reduction", policy,
      KOKKOS_LAMBDA(const std::size_t flat_index) {
        const std::size_t mode = flat_index / point_count;
        const std::size_t radial = flat_index - mode * point_count;
        const Complex derivative_psi = d42_first_derivative_at(
            &state(mode, psi_field, 0), point_count, radial, inverse_spacing);
        scratch(mode, dr_psi, radial) = derivative_psi;
        scratch(mode, effective_q, radial) =
            reduction == ReductionEvolution::StageConstrained
                ? derivative_psi
                : state(mode, q_field, radial);
      });

  Kokkos::parallel_for(
      "teuk_sbp_spatial_primitives", policy,
      KOKKOS_LAMBDA(const std::size_t flat_index) {
        const std::size_t mode = flat_index / point_count;
        const std::size_t radial = flat_index - mode * point_count;
        scratch(mode, dr_q, radial) = d42_first_derivative_at(
            &scratch(mode, effective_q, 0), point_count, radial,
            inverse_spacing);
        TeukolskyParameters parameters = base_parameters;
        parameters.azimuthal_mode = modes(mode);
        const TeukolskyCoefficients coefficients = teukolsky_coefficients(
            parameters, grid.coordinate(radial), theta);
        const TeukolskyState effective_state{
            state(mode, p_field, radial), scratch(mode, effective_q, radial),
            state(mode, psi_field, radial)};
        scratch(mode, psi_velocity, radial) =
            teukolsky_psi_rhs(coefficients, effective_state);
      });

  Kokkos::parallel_for(
      "teuk_sbp_linear_rhs", policy,
      KOKKOS_LAMBDA(const std::size_t flat_index) {
        const std::size_t mode = flat_index / point_count;
        const std::size_t radial = flat_index - mode * point_count;
        const Complex derivative_velocity = d42_first_derivative_at(
            &scratch(mode, psi_velocity, 0), point_count, radial,
            inverse_spacing);
        scratch(mode, dr_psi_velocity, radial) = derivative_velocity;

        TeukolskyParameters parameters = base_parameters;
        parameters.azimuthal_mode = modes(mode);
        const TeukolskyCoefficients coefficients = teukolsky_coefficients(
            parameters, grid.coordinate(radial), theta);
        const TeukolskyState effective_state{
            state(mode, p_field, radial), scratch(mode, effective_q, radial),
            state(mode, psi_field, radial)};
        const Complex constraint =
            state(mode, q_field, radial) - scratch(mode, dr_psi, radial);
        const double damping =
            reduction == ReductionEvolution::FreeDamped
                ? parameters.reduction_damping
                : 0.0;

        rhs(mode, p_field, radial) = teukolsky_p_rhs(
            coefficients, effective_state, scratch(mode, dr_q, radial),
            angular_laplacian(mode, radial), forcing(mode, radial));
        rhs(mode, q_field, radial) =
            teukolsky_q_rhs(derivative_velocity, constraint, damping);
        rhs(mode, psi_field, radial) =
            scratch(mode, psi_velocity, radial);

        if (dissipation_strength > 0.0) {
          rhs(mode, p_field, radial) += d42_compatible_dissipation_at(
              &state(mode, p_field, 0), point_count, radial, grid.spacing(),
              dissipation_strength);
          rhs(mode, q_field, radial) += d42_compatible_dissipation_at(
              &state(mode, q_field, 0), point_count, radial, grid.spacing(),
              dissipation_strength);
          rhs(mode, psi_field, radial) += d42_compatible_dissipation_at(
              &state(mode, psi_field, 0), point_count, radial, grid.spacing(),
              dissipation_strength);
        }
      });
}

// Existing owning-container convenience wrapper. It delegates to the explicit
// stage-View core and preserves the original API.
inline void evaluate_sbp_teukolsky_radial_lines_rhs(
    TeukolskyRadialLines& lines, const TeukolskyParameters& base_parameters,
    const double theta, const ReductionEvolution reduction,
    const double dissipation_strength = 0.0) {
  ExecutionSpace execution;
  evaluate_sbp_teukolsky_radial_lines_rhs(
      execution, lines.grid(), lines.modes(), lines.state(), lines.rhs(),
      lines.scratch(), lines.angular_laplacian(), lines.forcing(),
      base_parameters, theta, reduction, dissipation_strength);
}

}  // namespace teuk
