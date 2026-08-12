#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "teuk/background.hpp"
#include "teuk/fields.hpp"
#include "teuk/grid.hpp"
#include "teuk/linear_spatial.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/reconstruction.hpp"
#include "teuk/reconstruction_spatial.hpp"
#include "teuk/sbp.hpp"
#include "teuk/teukolsky.hpp"
#include "teuk/types.hpp"

namespace teuk {

using FullSpatialStateView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using FullSpatialValueView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using FullSpatialThetaView = Kokkos::View<double*, MemorySpace>;

struct TeukolskyFullFieldOffsets {
  std::size_t P = 0;
  std::size_t Q = 1;
  std::size_t Psi = 2;
};

struct ReconstructionFullFieldOffsets {
  std::size_t G = 0;
  std::size_t Lambda = 1;
  std::size_t H = 2;
  std::size_t B = 3;
  std::size_t Pi = 4;
  std::size_t C = 5;
  std::size_t U = 6;
};

KOKKOS_INLINE_FUNCTION
std::size_t maximum(const std::size_t a, const std::size_t b) {
  return a > b ? a : b;
}

KOKKOS_INLINE_FUNCTION
std::size_t maximum_field_offset(const TeukolskyFullFieldOffsets& offsets) {
  return maximum(offsets.P, maximum(offsets.Q, offsets.Psi));
}

KOKKOS_INLINE_FUNCTION
std::size_t maximum_field_offset(const ReconstructionFullFieldOffsets& o) {
  return maximum(maximum(maximum(o.G, o.Lambda), maximum(o.H, o.B)),
                 maximum(maximum(o.Pi, o.C), o.U));
}

namespace full_spatial_detail {

struct ViewStride4 {
  std::size_t mode;
  std::size_t field;
  std::size_t radial;
  std::size_t theta;
};

struct ViewStride3 {
  std::size_t mode;
  std::size_t radial;
  std::size_t theta;
};

template <class View>
ViewStride4 view_stride4(const View& view) {
  return {view.stride(0), view.stride(1), view.stride(2), view.stride(3)};
}

template <class View>
ViewStride3 view_stride3(const View& view) {
  return {view.stride(0), view.stride(1), view.stride(2)};
}

KOKKOS_INLINE_FUNCTION
std::size_t rank4_index(const std::size_t mode, const std::size_t field,
                        const std::size_t radial, const std::size_t theta,
                        const ViewStride4 stride) {
  return mode * stride.mode + field * stride.field +
         radial * stride.radial + theta * stride.theta;
}

KOKKOS_INLINE_FUNCTION
std::size_t rank3_index(const std::size_t mode, const std::size_t radial,
                        const std::size_t theta, const ViewStride3 stride) {
  return mode * stride.mode + radial * stride.radial + theta * stride.theta;
}

struct TeukolskyPrepareReductionFunctor {
  const Complex* state;
  Complex* scratch;
  ViewStride4 state_stride;
  ViewStride4 scratch_stride;
  std::size_t point_count;
  std::size_t theta_count;
  std::size_t psi_field;
  std::size_t q_field;
  double inverse_spacing;
  ReductionEvolution reduction;
  RadialDiscretization discretization;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t dr_psi = static_cast<std::size_t>(
        TeukolskyRadialScratch::RadialDerivativePsi);
    constexpr std::size_t effective_q =
        static_cast<std::size_t>(TeukolskyRadialScratch::EffectiveQ);
    const std::size_t mode_radial = flat / theta_count;
    const std::size_t theta = flat - mode_radial * theta_count;
    const std::size_t mode = mode_radial / point_count;
    const std::size_t radial = mode_radial - mode * point_count;
    const std::size_t psi_line =
        rank4_index(mode, psi_field, 0, theta, state_stride);
    const Complex derivative_psi = radial_first_derivative_strided_at(
        discretization, state + psi_line, point_count, radial,
        inverse_spacing, state_stride.radial);
    scratch[rank4_index(mode, dr_psi, radial, theta, scratch_stride)] =
        derivative_psi;
    scratch[rank4_index(mode, effective_q, radial, theta, scratch_stride)] =
        reduction == ReductionEvolution::StageConstrained
            ? derivative_psi
            : state[rank4_index(mode, q_field, radial, theta, state_stride)];
  }
};

struct TeukolskySpatialPrimitivesFunctor {
  const Complex* state;
  Complex* scratch;
  const int* signed_modes;
  const Real* theta_coordinates;
  UniformRadialGrid grid;
  TeukolskyParameters parameters;
  ViewStride4 state_stride;
  ViewStride4 scratch_stride;
  std::size_t point_count;
  std::size_t theta_count;
  TeukolskyFullFieldOffsets fields;
  double inverse_spacing;
  RadialDiscretization discretization;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t effective_q =
        static_cast<std::size_t>(TeukolskyRadialScratch::EffectiveQ);
    constexpr std::size_t dr_q =
        static_cast<std::size_t>(TeukolskyRadialScratch::RadialDerivativeQ);
    constexpr std::size_t psi_velocity =
        static_cast<std::size_t>(TeukolskyRadialScratch::PsiVelocity);
    const std::size_t mode_radial = flat / theta_count;
    const std::size_t theta = flat - mode_radial * theta_count;
    const std::size_t mode = mode_radial / point_count;
    const std::size_t radial = mode_radial - mode * point_count;
    const std::size_t q_line =
        rank4_index(mode, effective_q, 0, theta, scratch_stride);
    const Complex derivative_q = radial_first_derivative_strided_at(
        discretization, scratch + q_line, point_count, radial,
        inverse_spacing, scratch_stride.radial);
    scratch[rank4_index(mode, dr_q, radial, theta, scratch_stride)] =
        derivative_q;
    TeukolskyParameters point_parameters = parameters;
    point_parameters.azimuthal_mode = signed_modes[mode];
    const auto coefficients = teukolsky_coefficients(
        point_parameters, grid.coordinate(radial), theta_coordinates[theta]);
    const TeukolskyState point{
        state[rank4_index(mode, fields.P, radial, theta, state_stride)],
        scratch[
            rank4_index(mode, effective_q, radial, theta, scratch_stride)],
        state[rank4_index(mode, fields.Psi, radial, theta, state_stride)]};
    scratch[rank4_index(mode, psi_velocity, radial, theta, scratch_stride)] =
        teukolsky_psi_rhs(coefficients, point);
  }
};

struct TeukolskyRhsFunctor {
  const Complex* state;
  Complex* scratch;
  const Complex* angular_laplacian;
  const Complex* forcing;
  Complex* output;
  const int* signed_modes;
  const Real* theta_coordinates;
  UniformRadialGrid grid;
  TeukolskyParameters parameters;
  ViewStride4 state_stride;
  ViewStride4 scratch_stride;
  ViewStride3 angular_stride;
  ViewStride3 forcing_stride;
  ViewStride4 output_stride;
  std::size_t point_count;
  std::size_t theta_count;
  TeukolskyFullFieldOffsets stage_fields;
  TeukolskyFullFieldOffsets output_fields;
  ReductionEvolution reduction;
  double inverse_spacing;
  double dissipation_strength;
  RadialDiscretization discretization;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t dr_psi = static_cast<std::size_t>(
        TeukolskyRadialScratch::RadialDerivativePsi);
    constexpr std::size_t effective_q =
        static_cast<std::size_t>(TeukolskyRadialScratch::EffectiveQ);
    constexpr std::size_t dr_q =
        static_cast<std::size_t>(TeukolskyRadialScratch::RadialDerivativeQ);
    constexpr std::size_t psi_velocity =
        static_cast<std::size_t>(TeukolskyRadialScratch::PsiVelocity);
    constexpr std::size_t dr_psi_velocity = static_cast<std::size_t>(
        TeukolskyRadialScratch::RadialDerivativePsiVelocity);
    const std::size_t mode_radial = flat / theta_count;
    const std::size_t theta = flat - mode_radial * theta_count;
    const std::size_t mode = mode_radial / point_count;
    const std::size_t radial = mode_radial - mode * point_count;
    const std::size_t velocity_line =
        rank4_index(mode, psi_velocity, 0, theta, scratch_stride);
    const Complex derivative_velocity = radial_first_derivative_strided_at(
        discretization, scratch + velocity_line, point_count, radial,
        inverse_spacing, scratch_stride.radial);
    scratch[
        rank4_index(mode, dr_psi_velocity, radial, theta, scratch_stride)] =
        derivative_velocity;
    TeukolskyParameters point_parameters = parameters;
    point_parameters.azimuthal_mode = signed_modes[mode];
    const auto coefficients = teukolsky_coefficients(
        point_parameters, grid.coordinate(radial), theta_coordinates[theta]);
    const TeukolskyState point{
        state[rank4_index(mode, stage_fields.P, radial, theta, state_stride)],
        scratch[
            rank4_index(mode, effective_q, radial, theta, scratch_stride)],
        state[
            rank4_index(mode, stage_fields.Psi, radial, theta, state_stride)]};
    const Complex constraint =
        state[rank4_index(mode, stage_fields.Q, radial, theta, state_stride)] -
        scratch[rank4_index(mode, dr_psi, radial, theta, scratch_stride)];
    const double damping = reduction == ReductionEvolution::FreeDamped
                               ? parameters.reduction_damping
                               : 0.0;
    const std::size_t angular_index =
        rank3_index(mode, radial, theta, angular_stride);
    const std::size_t forcing_index =
        rank3_index(mode, radial, theta, forcing_stride);
    const std::size_t p_output =
        rank4_index(mode, output_fields.P, radial, theta, output_stride);
    const std::size_t q_output =
        rank4_index(mode, output_fields.Q, radial, theta, output_stride);
    const std::size_t psi_output =
        rank4_index(mode, output_fields.Psi, radial, theta, output_stride);
    output[p_output] = teukolsky_p_rhs(
        coefficients, point,
        scratch[rank4_index(mode, dr_q, radial, theta, scratch_stride)],
        angular_laplacian[angular_index], forcing[forcing_index]);
    output[q_output] =
        teukolsky_q_rhs(derivative_velocity, constraint, damping);
    output[psi_output] =
        scratch[
            rank4_index(mode, psi_velocity, radial, theta, scratch_stride)];
    if (dissipation_strength > 0.0) {
      const auto dissipation = [&](const std::size_t field) {
        const std::size_t line =
            rank4_index(mode, field, 0, theta, state_stride);
        return radial_compatible_dissipation_at(
            discretization, state + line, point_count, radial,
            grid.spacing(), dissipation_strength, state_stride.radial);
      };
      output[p_output] += dissipation(stage_fields.P);
      output[q_output] += dissipation(stage_fields.Q);
      output[psi_output] += dissipation(stage_fields.Psi);
    }
  }
};

static_assert(std::is_trivially_copyable_v<TeukolskyPrepareReductionFunctor>);
static_assert(std::is_trivially_copyable_v<TeukolskySpatialPrimitivesFunctor>);
static_assert(std::is_trivially_copyable_v<TeukolskyRhsFunctor>);
static_assert(sizeof(TeukolskyPrepareReductionFunctor) < 1800);
static_assert(sizeof(TeukolskySpatialPrimitivesFunctor) < 1800);
static_assert(sizeof(TeukolskyRhsFunctor) < 1800);

}  // namespace full_spatial_detail

template <class View4D>
KOKKOS_INLINE_FUNCTION Complex full_strided_radial_derivative_at(
    const View4D& values, const std::size_t mode, const std::size_t field,
    const std::size_t point_count, const std::size_t radial,
    const std::size_t theta, const double inverse_spacing) {
  return d42_first_derivative_strided_at(
      &values(mode, field, 0, theta), point_count, radial, inverse_spacing,
      values.extent(3));
}

template <class View4D>
KOKKOS_INLINE_FUNCTION Complex full_strided_dissipation_at(
    const View4D& values, const std::size_t mode, const std::size_t field,
    const std::size_t point_count, const std::size_t radial,
    const std::size_t theta, const double spacing, const double strength) {
  const std::size_t first_row = radial > 3 ? radial - 3 : 0;
  const std::size_t last_row =
      radial < point_count - 3 ? radial : point_count - 4;
  const std::size_t stride = values.extent(3);
  const Complex* line = &values(mode, field, 0, theta);
  Complex normal_product = 0.0;
  for (std::size_t row = first_row; row <= last_row; ++row) {
    const Complex difference =
        -line[row * stride] + 3.0 * line[(row + 1) * stride] -
        3.0 * line[(row + 2) * stride] + line[(row + 3) * stride];
    const std::size_t position = radial - row;
    const double transpose =
        position == 0 ? -1.0 : (position == 1 ? 3.0
                                               : (position == 2 ? -3.0 : 1.0));
    normal_product += transpose * difference;
  }
  return -strength * normal_product /
         (spacing * d42_norm_weight(point_count, radial));
}

template <class CallerExecutionSpace, class ThetaView, class ModeView,
          class StageView, class AngularView, class ForcingView,
          class ScratchView, class OutputView>
inline void evaluate_sbp_teukolsky_full_stage_rhs(
    const CallerExecutionSpace& execution, const UniformRadialGrid& grid,
    const TeukolskyParameters& base_parameters,
    const ThetaView& theta_coordinates, const ModeView& signed_modes,
    const StageView& stage_state, const AngularView& angular_laplacian,
    const ForcingView& forcing, const ReductionEvolution reduction,
    const ScratchView& scratch, const OutputView& output_rhs,
    const double dissipation_strength = 0.0,
    const TeukolskyFullFieldOffsets stage_fields = {},
    const TeukolskyFullFieldOffsets output_fields = {},
    const RadialDiscretization discretization = RadialDiscretization::D42) {
  static_assert(StageView::rank == 4 && ScratchView::rank == 4 &&
                    OutputView::rank == 4 && AngularView::rank == 3 &&
                    ForcingView::rank == 3,
                "full Teukolsky kernels require rank-4 state and rank-3 values");
  const std::size_t mode_count = stage_state.extent(0);
  const std::size_t point_count = grid.size();
  const std::size_t theta_count = stage_state.extent(3);
  if (point_count < radial_minimum_points(discretization) ||
      stage_state.extent(1) <= maximum_field_offset(stage_fields) ||
      stage_state.extent(2) != point_count || signed_modes.extent(0) != mode_count ||
      theta_coordinates.extent(0) != theta_count ||
      angular_laplacian.extent(0) != mode_count ||
      angular_laplacian.extent(1) != point_count ||
      angular_laplacian.extent(2) != theta_count ||
      forcing.extent(0) != mode_count || forcing.extent(1) != point_count ||
      forcing.extent(2) != theta_count || output_rhs.extent(0) != mode_count ||
      output_rhs.extent(1) <= maximum_field_offset(output_fields) ||
      output_rhs.extent(2) != point_count ||
      output_rhs.extent(3) != theta_count || scratch.extent(0) != mode_count ||
      scratch.extent(1) !=
          static_cast<std::size_t>(TeukolskyRadialScratch::Count) ||
      scratch.extent(2) != point_count || scratch.extent(3) != theta_count) {
    throw std::invalid_argument("full Teukolsky stage view extents mismatch");
  }
  if (!(base_parameters.compactification_length > 0.0)) {
    throw std::invalid_argument("compactification length must be positive");
  }
  if (dissipation_strength < 0.0) {
    throw std::invalid_argument("dissipation strength must be nonnegative");
  }

  const double inverse_spacing = 1.0 / grid.spacing();
  const std::size_t total_points = mode_count * point_count * theta_count;
  const Kokkos::RangePolicy<CallerExecutionSpace> policy(
      execution, 0,
      static_cast<typename CallerExecutionSpace::size_type>(total_points));

  Kokkos::parallel_for(
      "teuk_full_prepare_reduction", policy,
      full_spatial_detail::TeukolskyPrepareReductionFunctor{
          stage_state.data(), scratch.data(),
          full_spatial_detail::view_stride4(stage_state),
          full_spatial_detail::view_stride4(scratch),
          point_count, theta_count, stage_fields.Psi, stage_fields.Q,
          inverse_spacing, reduction, discretization});

  Kokkos::parallel_for(
      "teuk_full_spatial_primitives", policy,
      full_spatial_detail::TeukolskySpatialPrimitivesFunctor{
          stage_state.data(), scratch.data(), signed_modes.data(),
          theta_coordinates.data(), grid, base_parameters,
          full_spatial_detail::view_stride4(stage_state),
          full_spatial_detail::view_stride4(scratch), point_count, theta_count,
          stage_fields, inverse_spacing, discretization});

  Kokkos::parallel_for(
      "teuk_full_rhs", policy, full_spatial_detail::TeukolskyRhsFunctor{
                                   stage_state.data(),
                                   scratch.data(),
                                   angular_laplacian.data(),
                                   forcing.data(),
                                   output_rhs.data(),
                                   signed_modes.data(),
                                   theta_coordinates.data(),
                                   grid,
                                   base_parameters,
                                   full_spatial_detail::view_stride4(
                                       stage_state),
                                   full_spatial_detail::view_stride4(scratch),
                                   full_spatial_detail::view_stride3(
                                       angular_laplacian),
                                   full_spatial_detail::view_stride3(forcing),
                                   full_spatial_detail::view_stride4(output_rhs),
                                   point_count,
                                   theta_count,
                                   stage_fields,
                                   output_fields,
                                   reduction,
                                   inverse_spacing,
                                   dissipation_strength,
                                   discretization});
}

template <class CallerExecutionSpace, class StageView, class DerivativeView>
inline void evaluate_sbp_reconstruction_full_radial_derivatives(
    const CallerExecutionSpace& execution, const UniformRadialGrid& grid,
    const StageView& stage_state,
    const DerivativeView& output_radial_derivatives,
    const ReconstructionFullFieldOffsets stage_fields = {},
    const ReconstructionFullFieldOffsets derivative_fields = {}) {
  static_assert(StageView::rank == 4 && DerivativeView::rank == 4,
                "full reconstruction derivatives require rank-4 views");
  const std::size_t mode_count = stage_state.extent(0);
  constexpr std::size_t field_count = 7;
  const std::size_t point_count = grid.size();
  const std::size_t theta_count = stage_state.extent(3);
  if (point_count < d42_minimum_points ||
      stage_state.extent(1) <= maximum_field_offset(stage_fields) ||
      stage_state.extent(2) != point_count ||
      output_radial_derivatives.extent(0) != mode_count ||
      output_radial_derivatives.extent(1) <=
          maximum_field_offset(derivative_fields) ||
      output_radial_derivatives.extent(2) != point_count ||
      output_radial_derivatives.extent(3) != theta_count) {
    throw std::invalid_argument(
        "full reconstruction radial derivative view extents mismatch");
  }
  const double inverse_spacing = 1.0 / grid.spacing();
  const std::size_t total = mode_count * field_count * point_count * theta_count;
  const Kokkos::RangePolicy<CallerExecutionSpace> policy(
      execution, 0,
      static_cast<typename CallerExecutionSpace::size_type>(total));
  Kokkos::parallel_for(
      "teuk_full_reconstruction_radial_derivatives", policy,
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_field_radial = flat / theta_count;
        const std::size_t theta = flat - mode_field_radial * theta_count;
        const std::size_t mode_field = mode_field_radial / point_count;
        const std::size_t radial =
            mode_field_radial - mode_field * point_count;
        const std::size_t mode = mode_field / field_count;
        const std::size_t logical_field = mode_field - mode * field_count;
        const std::size_t input_offsets[7] = {
            stage_fields.G,  stage_fields.Lambda, stage_fields.H,
            stage_fields.B,  stage_fields.Pi,     stage_fields.C,
            stage_fields.U};
        const std::size_t output_offsets[7] = {
            derivative_fields.G,  derivative_fields.Lambda,
            derivative_fields.H,  derivative_fields.B,
            derivative_fields.Pi, derivative_fields.C,
            derivative_fields.U};
        output_radial_derivatives(mode, output_offsets[logical_field], radial,
                                  theta) =
            full_strided_radial_derivative_at(
                stage_state, mode, input_offsets[logical_field], point_count,
                radial, theta, inverse_spacing);
      });
}

template <class ThetaView, class SharpView, class StageView, class Psi4View,
          class OutputView, class DerivativeView>
inline void validate_reconstruction_full_stage_views(
    const UniformRadialGrid& grid, const KerrParameters& parameters,
    const ThetaView& theta_coordinates, const SharpView& sharp_indices,
    const StageView& stage_state, const Psi4View& stage_psi4,
    const OutputView& output_rhs, const DerivativeView& radial_derivatives,
    const ReconstructionFullFieldOffsets& stage_fields,
    const ReconstructionFullFieldOffsets& output_fields,
    const ReconstructionFullFieldOffsets& derivative_fields) {
  static_assert(StageView::rank == 4 && Psi4View::rank == 3 &&
                    OutputView::rank == 4 && DerivativeView::rank == 4,
                "full reconstruction kernels require rank-4 fields and rank-3 Psi4");
  const std::size_t modes = stage_state.extent(0);
  const std::size_t radial = grid.size();
  const std::size_t theta = stage_state.extent(3);
  if (!(parameters.compactification_length > 0.0) || radial < d42_minimum_points ||
      stage_state.extent(1) <= maximum_field_offset(stage_fields) ||
      stage_state.extent(2) != radial ||
      theta_coordinates.extent(0) != theta || sharp_indices.extent(0) != modes ||
      stage_psi4.extent(0) != modes || stage_psi4.extent(1) != radial ||
      stage_psi4.extent(2) != theta || output_rhs.extent(0) != modes ||
      output_rhs.extent(1) <= maximum_field_offset(output_fields) ||
      output_rhs.extent(2) != radial ||
      output_rhs.extent(3) != theta || radial_derivatives.extent(0) != modes ||
      radial_derivatives.extent(1) <= maximum_field_offset(derivative_fields) ||
      radial_derivatives.extent(2) != radial ||
      radial_derivatives.extent(3) != theta) {
    throw std::invalid_argument("full reconstruction stage view extents mismatch");
  }
}

template <class CallerExecutionSpace, class ThetaView, class SharpView,
          class StageView, class Psi4View, class AngularView,
          class DerivativeView, class OutputView>
inline void evaluate_sbp_reconstruction_full_pass1(
    const CallerExecutionSpace& execution, const UniformRadialGrid& grid,
    const KerrParameters& parameters, const ThetaView& theta_coordinates,
    const SharpView& sharp_indices, const StageView& stage_state,
    const Psi4View& stage_psi4, const AngularView& eth1_f,
    const DerivativeView& radial_dr, const OutputView& output_rhs,
    const ReconstructionFullFieldOffsets stage_fields = {},
    const ReconstructionFullFieldOffsets derivative_fields = {},
    const ReconstructionFullFieldOffsets output_fields = {}) {
  validate_reconstruction_full_stage_views(
      grid, parameters, theta_coordinates, sharp_indices, stage_state,
      stage_psi4, output_rhs, radial_dr, stage_fields, output_fields,
      derivative_fields);
  const std::size_t modes = stage_state.extent(0);
  const std::size_t radial_count = grid.size();
  const std::size_t theta_count = stage_state.extent(3);
  if (eth1_f.extent(0) != modes || eth1_f.extent(1) != radial_count ||
      eth1_f.extent(2) != theta_count) {
    throw std::invalid_argument("reconstruction pass1 angular extents mismatch");
  }
  const std::size_t total = modes * radial_count * theta_count;
  const Kokkos::RangePolicy<CallerExecutionSpace> policy(
      execution, 0,
      static_cast<typename CallerExecutionSpace::size_type>(total));
  Kokkos::parallel_for(
      "teuk_full_reconstruction_pass1", policy,
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_radial = flat / theta_count;
        const std::size_t theta = flat - mode_radial * theta_count;
        const std::size_t mode = mode_radial / radial_count;
        const std::size_t radial = mode_radial - mode * radial_count;
        const double radius = grid.coordinate(radial);
        const auto background = kerr_background_point(
            parameters, radius, theta_coordinates(theta));
        const Complex g = stage_state(mode, stage_fields.G, radial, theta);
        const Complex lambda =
            stage_state(mode, stage_fields.Lambda, radial, theta);
        const Complex f = stage_psi4(mode, radial, theta);
        const Complex delta_g = -4.0 * radius * background.mu0 * g +
                                eth1_f(mode, radial, theta) -
                                radius * background.tau0 * f;
        const Complex delta_lambda =
            -radius * (background.mu0 + Kokkos::conj(background.mu0)) *
                lambda -
            f;
        output_rhs(mode, output_fields.G, radial, theta) =
            reconstruction_time_derivative(
            g, radial_dr(mode, derivative_fields.G, radial, theta), delta_g, 2, radius,
            parameters.mass, parameters.compactification_length);
        output_rhs(mode, output_fields.Lambda, radial, theta) =
            reconstruction_time_derivative(
                lambda, radial_dr(mode, derivative_fields.Lambda, radial, theta), delta_lambda, 1,
                radius, parameters.mass, parameters.compactification_length);
      });
}

template <class CallerExecutionSpace, class ThetaView, class SharpView,
          class StageView, class Psi4View, class AngularView,
          class DerivativeView, class OutputView>
inline void evaluate_sbp_reconstruction_full_pass2(
    const CallerExecutionSpace& execution, const UniformRadialGrid& grid,
    const KerrParameters& parameters, const ThetaView& theta_coordinates,
    const SharpView& sharp_indices, const StageView& stage_state,
    const Psi4View& stage_psi4, const AngularView& eth2_g,
    const DerivativeView& radial_dr, const OutputView& output_rhs,
    const ReconstructionFullFieldOffsets stage_fields = {},
    const ReconstructionFullFieldOffsets derivative_fields = {},
    const ReconstructionFullFieldOffsets output_fields = {}) {
  validate_reconstruction_full_stage_views(
      grid, parameters, theta_coordinates, sharp_indices, stage_state,
      stage_psi4, output_rhs, radial_dr, stage_fields, output_fields,
      derivative_fields);
  const std::size_t modes = stage_state.extent(0);
  const std::size_t radial_count = grid.size();
  const std::size_t theta_count = stage_state.extent(3);
  if (eth2_g.extent(0) != modes || eth2_g.extent(1) != radial_count ||
      eth2_g.extent(2) != theta_count) {
    throw std::invalid_argument("reconstruction pass2 angular extents mismatch");
  }
  const std::size_t total = modes * radial_count * theta_count;
  const Kokkos::RangePolicy<CallerExecutionSpace> policy(
      execution, 0,
      static_cast<typename CallerExecutionSpace::size_type>(total));
  Kokkos::parallel_for(
      "teuk_full_reconstruction_pass2", policy,
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_radial = flat / theta_count;
        const std::size_t theta = flat - mode_radial * theta_count;
        const std::size_t mode = mode_radial / radial_count;
        const std::size_t radial = mode_radial - mode * radial_count;
        const double radius = grid.coordinate(radial);
        const double radius2 = radius * radius;
        const auto background = kerr_background_point(
            parameters, radius, theta_coordinates(theta));
        const Complex mu_bar = Kokkos::conj(background.mu0);
        const Complex pi_bar = Kokkos::conj(background.pi0);
        const Complex g = stage_state(mode, stage_fields.G, radial, theta);
        const Complex lambda =
            stage_state(mode, stage_fields.Lambda, radial, theta);
        const Complex h = stage_state(mode, stage_fields.H, radial, theta);
        const Complex b = stage_state(mode, stage_fields.B, radial, theta);
        const Complex pi = stage_state(mode, stage_fields.Pi, radial, theta);
        const Complex c = stage_state(mode, stage_fields.C, radial, theta);
        const Complex delta_h =
            -3.0 * radius * background.mu0 * h +
            eth2_g(mode, radial, theta) -
            2.0 * radius * background.tau0 * g;
        const Complex delta_b =
            radius * (background.mu0 - mu_bar) * b - 2.0 * lambda;
        const Complex delta_pi =
            -g - radius * (pi_bar + background.tau0) * lambda +
            0.5 * radius2 * background.mu0 *
                (pi_bar + background.tau0) * b;
        const Complex delta_c = -radius * mu_bar * c - 2.0 * pi -
                                radius * background.tau0 * b;
        output_rhs(mode, output_fields.H, radial, theta) = reconstruction_time_derivative(
            h, radial_dr(mode, derivative_fields.H, radial, theta), delta_h, 3, radius,
            parameters.mass, parameters.compactification_length);
        output_rhs(mode, output_fields.B, radial, theta) = reconstruction_time_derivative(
            b, radial_dr(mode, derivative_fields.B, radial, theta), delta_b, 1, radius,
            parameters.mass, parameters.compactification_length);
        output_rhs(mode, output_fields.Pi, radial, theta) = reconstruction_time_derivative(
            pi, radial_dr(mode, derivative_fields.Pi, radial, theta), delta_pi, 2, radius,
            parameters.mass, parameters.compactification_length);
        output_rhs(mode, output_fields.C, radial, theta) = reconstruction_time_derivative(
            c, radial_dr(mode, derivative_fields.C, radial, theta), delta_c, 2, radius,
            parameters.mass, parameters.compactification_length);
      });
}

template <class CallerExecutionSpace, class ThetaView, class SharpView,
          class StageView, class Psi4View, class AngularView,
          class DerivativeView, class OutputView>
inline void evaluate_sbp_reconstruction_full_pass3(
    const CallerExecutionSpace& execution, const UniformRadialGrid& grid,
    const KerrParameters& parameters, const ThetaView& theta_coordinates,
    const SharpView& sharp_indices, const StageView& stage_state,
    const Psi4View& stage_psi4, const AngularView& stage_angular,
    const DerivativeView& radial_dr, const OutputView& output_rhs,
    const ReconstructionFullFieldOffsets stage_fields = {},
    const ReconstructionFullFieldOffsets derivative_fields = {},
    const ReconstructionFullFieldOffsets output_fields = {}) {
  validate_reconstruction_full_stage_views(
      grid, parameters, theta_coordinates, sharp_indices, stage_state,
      stage_psi4, output_rhs, radial_dr, stage_fields, output_fields,
      derivative_fields);
  const std::size_t modes = stage_state.extent(0);
  const std::size_t radial_count = grid.size();
  const std::size_t theta_count = stage_state.extent(3);
  if (stage_angular.extent(0) != modes ||
      stage_angular.extent(1) !=
          static_cast<std::size_t>(ReconstructionAngularInput::Count) ||
      stage_angular.extent(2) != radial_count ||
      stage_angular.extent(3) != theta_count) {
    throw std::invalid_argument("reconstruction pass3 angular extents mismatch");
  }
  const std::size_t total = modes * radial_count * theta_count;
  const Kokkos::RangePolicy<CallerExecutionSpace> policy(
      execution, 0,
      static_cast<typename CallerExecutionSpace::size_type>(total));
  Kokkos::parallel_for(
      "teuk_full_reconstruction_pass3", policy,
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_radial = flat / theta_count;
        const std::size_t theta = flat - mode_radial * theta_count;
        const std::size_t mode = mode_radial / radial_count;
        const std::size_t radial = mode_radial - mode * radial_count;
        const std::size_t sharp_mode = sharp_indices(mode);
        const double radius = grid.coordinate(radial);
        const auto background = kerr_background_point(
            parameters, radius, theta_coordinates(theta));
        const ReconstructionFields fields{
            stage_psi4(mode, radial, theta),
            stage_state(mode, stage_fields.G, radial, theta),
            stage_state(mode, stage_fields.H, radial, theta),
            stage_state(mode, stage_fields.Lambda, radial, theta),
            stage_state(mode, stage_fields.Pi, radial, theta),
            stage_state(mode, stage_fields.B, radial, theta),
            stage_state(mode, stage_fields.C, radial, theta),
            stage_state(mode, stage_fields.U, radial, theta),
            Kokkos::conj(stage_state(sharp_mode, stage_fields.Pi, radial, theta)),
            Kokkos::conj(stage_state(sharp_mode, stage_fields.B, radial, theta)),
            Kokkos::conj(stage_state(sharp_mode, stage_fields.C, radial, theta))};
        const ReconstructionAngularDerivatives angular{
            stage_angular(mode, 0, radial, theta),
            stage_angular(mode, 1, radial, theta),
            stage_angular(mode, 2, radial, theta),
            stage_angular(mode, 3, radial, theta),
            stage_angular(mode, 4, radial, theta),
            stage_angular(mode, 5, radial, theta)};
        const auto delta =
            reconstruction_delta_rhs(radius, background, fields, angular);
        output_rhs(mode, output_fields.U, radial, theta) = reconstruction_time_derivative(
            fields.U, radial_dr(mode, derivative_fields.U, radial, theta), delta.U, 3, radius,
            parameters.mass, parameters.compactification_length);
      });
}

}  // namespace teuk
