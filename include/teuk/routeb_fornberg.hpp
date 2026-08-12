#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "teuk/routeb_fornberg_weights.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Direct uniform-grid derivative.  Each derivative order has its own
// independently generated nine-point formula; this function never applies a
// first-derivative operator recursively.  The window is centered where
// possible and shifted intact to either boundary.
template <class Value>
KOKKOS_INLINE_FUNCTION Value routeb_fornberg_direct_derivative_at(
    const int derivative_order, const Value* values,
    const std::size_t point_count, const std::size_t index,
    const double inverse_spacing, const std::size_t stride = 1) {
  if (derivative_order < 1 || derivative_order > 4 ||
      point_count < routeb_fornberg_window || index >= point_count) {
    return Value{};
  }
  constexpr std::size_t center = routeb_fornberg_window / 2;
  const std::size_t maximum_start = point_count - routeb_fornberg_window;
  const std::size_t proposed_start = index > center ? index - center : 0;
  const std::size_t start =
      proposed_start < maximum_start ? proposed_start : maximum_start;
  const std::size_t local_row = index - start;
  Value result{};
  for (std::size_t column = 0; column < routeb_fornberg_window; ++column) {
    result += routeb_fornberg_weight(
                  static_cast<std::size_t>(derivative_order - 1), local_row,
                  column) *
              values[(start + column) * stride];
  }
  double scale = 1.0;
  for (int power = 0; power < derivative_order; ++power) {
    scale *= inverse_spacing;
  }
  return scale * result;
}

namespace routeb_fornberg_detail {

struct RankFourStrides {
  std::size_t mode;
  std::size_t field;
  std::size_t radial;
  std::size_t theta;
};

struct RankFiveStrides {
  std::size_t mode;
  std::size_t derivative;
  std::size_t field;
  std::size_t radial;
  std::size_t theta;
};

template <class Value>
struct DirectDerivativeFunctor {
  const Value* input;
  Value* output;
  RankFourStrides input_stride;
  RankFiveStrides output_stride;
  std::size_t field_count;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t radial_theta = radial_count * theta_count;
    const std::size_t field_radial_theta = field_count * radial_theta;
    const std::size_t mode = flat / field_radial_theta;
    const std::size_t within_mode = flat - mode * field_radial_theta;
    const std::size_t field = within_mode / radial_theta;
    const std::size_t within_field = within_mode - field * radial_theta;
    const std::size_t radial = within_field / theta_count;
    const std::size_t theta = within_field - radial * theta_count;
    const std::size_t input_base = mode * input_stride.mode +
                                   field * input_stride.field +
                                   theta * input_stride.theta;
    for (int derivative = 1; derivative <= 4; ++derivative) {
      const std::size_t output_index =
          mode * output_stride.mode +
          static_cast<std::size_t>(derivative - 1) *
              output_stride.derivative +
          field * output_stride.field + radial * output_stride.radial +
          theta * output_stride.theta;
      output[output_index] = routeb_fornberg_direct_derivative_at(
          derivative, input + input_base, radial_count, radial,
          inverse_spacing, input_stride.radial);
    }
  }
};

template <class LeftView, class RightView>
bool allocations_overlap(const LeftView& left, const RightView& right) {
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
  const auto left_end =
      left_begin + left.span() * sizeof(typename LeftView::value_type);
  const auto right_end =
      right_begin + right.span() * sizeof(typename RightView::value_type);
  return left_begin < right_end && right_begin < left_end;
}

struct StrideExtent {
  std::size_t stride;
  std::size_t extent;
};

// Accept packed or padded lexicographic layouts in any dimension order.  This
// sufficient separability condition deliberately rejects exotic injective
// affine mappings rather than risk internally aliased reads or parallel
// writes from a malformed LayoutStride.
template <std::size_t Rank, class View>
bool has_separated_strides(const View& view) {
  std::array<StrideExtent, Rank> dimensions{};
  for (std::size_t dimension = 0; dimension < Rank; ++dimension) {
    dimensions[dimension] =
        StrideExtent{view.stride(dimension), view.extent(dimension)};
    if (dimensions[dimension].stride == 0) return false;
  }
  for (std::size_t index = 1; index < Rank; ++index) {
    const StrideExtent value = dimensions[index];
    std::size_t insertion = index;
    while (insertion > 0 &&
           dimensions[insertion - 1].stride > value.stride) {
      dimensions[insertion] = dimensions[insertion - 1];
      --insertion;
    }
    dimensions[insertion] = value;
  }
  std::size_t occupied_span = 1;
  for (const StrideExtent dimension : dimensions) {
    if (dimension.extent <= 1) continue;
    if (dimension.stride < occupied_span ||
        dimension.extent - 1 >
            (std::numeric_limits<std::size_t>::max() - occupied_span) /
                dimension.stride) {
      return false;
    }
    occupied_span += (dimension.extent - 1) * dimension.stride;
  }
  return true;
}

}  // namespace routeb_fornberg_detail

// Apply all four direct operators to a caller-owned rank-four field and write
// (mode, derivative-1, field, radial, theta).  Both Views may be LayoutRight
// or LayoutStride.  Evaluation launches once, allocates nothing, and fences
// nowhere.
template <class ExecutionSpace, class InputView, class OutputView>
void evaluate_routeb_fornberg_derivatives(
    const ExecutionSpace& execution, const InputView& input,
    const OutputView& output, const double spacing) {
  static_assert(InputView::rank == 4 && OutputView::rank == 5,
                "Route-B Fornberg fields require rank four and rank five");
  using Value = typename InputView::non_const_value_type;
  static_assert(
      std::is_same_v<Value, typename OutputView::non_const_value_type>,
      "Route-B Fornberg input and output values must match");
  static_assert(!std::is_const_v<typename OutputView::value_type>,
                "Route-B Fornberg output must be writable");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename InputView::memory_space>::accessible &&
          Kokkos::SpaceAccessibility<
              ExecutionSpace, typename OutputView::memory_space>::accessible,
      "Route-B Fornberg views must be accessible from the execution space");
  const double inverse_spacing = 1.0 / spacing;
  const double inverse_spacing_squared = inverse_spacing * inverse_spacing;
  const double inverse_spacing_fourth =
      inverse_spacing_squared * inverse_spacing_squared;
  if (!std::isfinite(spacing) || !(spacing > 0.0) ||
      !std::isfinite(inverse_spacing_fourth) || input.data() == nullptr ||
      output.data() == nullptr || input.extent(0) == 0 ||
      input.extent(1) == 0 || input.extent(3) == 0 ||
      input.extent(2) < routeb_fornberg_window ||
      output.extent(0) != input.extent(0) ||
      output.extent(1) != routeb_fornberg_derivative_count ||
      output.extent(2) != input.extent(1) ||
      output.extent(3) != input.extent(2) ||
      output.extent(4) != input.extent(3) ||
      !routeb_fornberg_detail::has_separated_strides<4>(input) ||
      !routeb_fornberg_detail::has_separated_strides<5>(output)) {
    throw std::invalid_argument("Route-B Fornberg derivative extents are invalid");
  }
  if (routeb_fornberg_detail::allocations_overlap(input, output)) {
    throw std::invalid_argument("Route-B Fornberg input output alias");
  }
  const std::size_t total = input.extent(0) * input.extent(1) *
                            input.extent(2) * input.extent(3);
  using Functor = routeb_fornberg_detail::DirectDerivativeFunctor<Value>;
  static_assert(std::is_trivially_copyable_v<Functor>);
  const routeb_fornberg_detail::RankFourStrides input_stride{
      input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
  const routeb_fornberg_detail::RankFiveStrides output_stride{
      output.stride(0), output.stride(1), output.stride(2), output.stride(3),
      output.stride(4)};
  Kokkos::parallel_for(
      "routeb_direct_fornberg_d1_d4",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
      Functor{input.data(), output.data(), input_stride, output_stride,
              input.extent(1), input.extent(2), input.extent(3),
              inverse_spacing});
}

}  // namespace teuk
