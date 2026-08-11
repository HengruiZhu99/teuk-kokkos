#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/grid.hpp"

namespace teuk {

inline constexpr std::size_t d42_minimum_points = 8;

// Dimensionless diagonal norm weight, H_ii / h, for the classical diagonal-
// norm D4-2 SBP first derivative.  The norm is symmetric about the grid.
KOKKOS_INLINE_FUNCTION double d42_norm_weight(
    const std::size_t point_count, const std::size_t index) {
  const std::size_t distance_from_boundary =
      index < point_count - 1 - index ? index : point_count - 1 - index;
  if (distance_from_boundary == 0) return 17.0 / 48.0;
  if (distance_from_boundary == 1) return 59.0 / 48.0;
  if (distance_from_boundary == 2) return 43.0 / 48.0;
  if (distance_from_boundary == 3) return 49.0 / 48.0;
  return 1.0;
}

// Dimensionless entry h*D_(row,column).  The first four rows are written out
// explicitly; the last four follow from odd reflection.  Interior rows use
// the fourth-order centered stencil.  Zero is returned outside the compact
// stencil.
KOKKOS_INLINE_FUNCTION double d42_derivative_coefficient(
    const std::size_t point_count, const std::size_t row,
    const std::size_t column) {
  // Express the odd-reflected right closure without recursion: SYCL device
  // compilation rejects even statically bounded recursive call graphs.
  const bool reflected = row + 4 >= point_count;
  const std::size_t local_row = reflected ? point_count - 1 - row : row;
  const std::size_t local_column =
      reflected ? point_count - 1 - column : column;
  double coefficient = 0.0;

  if (local_row == 0) {
    if (local_column == 0) coefficient = -24.0 / 17.0;
    else if (local_column == 1) coefficient = 59.0 / 34.0;
    else if (local_column == 2) coefficient = -4.0 / 17.0;
    else if (local_column == 3) coefficient = -3.0 / 34.0;
    return reflected ? -coefficient : coefficient;
  }
  if (local_row == 1) {
    if (local_column == 0) coefficient = -0.5;
    else if (local_column == 2) coefficient = 0.5;
    return reflected ? -coefficient : coefficient;
  }
  if (local_row == 2) {
    if (local_column == 0) coefficient = 4.0 / 43.0;
    else if (local_column == 1) coefficient = -59.0 / 86.0;
    else if (local_column == 3) coefficient = 59.0 / 86.0;
    else if (local_column == 4) coefficient = -4.0 / 43.0;
    return reflected ? -coefficient : coefficient;
  }
  if (local_row == 3) {
    if (local_column == 0) coefficient = 3.0 / 98.0;
    else if (local_column == 2) coefficient = -59.0 / 98.0;
    else if (local_column == 4) coefficient = 32.0 / 49.0;
    else if (local_column == 5) coefficient = -4.0 / 49.0;
    return reflected ? -coefficient : coefficient;
  }

  if (local_column + 2 == local_row) return 1.0 / 12.0;
  if (local_column + 1 == local_row) return -2.0 / 3.0;
  if (local_column == local_row + 1) return 2.0 / 3.0;
  if (local_column == local_row + 2) return -1.0 / 12.0;
  return 0.0;
}

KOKKOS_INLINE_FUNCTION double d42_norm_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return row == column ? grid.spacing() * d42_norm_weight(grid.size(), row)
                       : 0.0;
}

KOKKOS_INLINE_FUNCTION double d42_derivative_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return d42_derivative_coefficient(grid.size(), row, column) /
         grid.spacing();
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value d42_first_derivative_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t row, const double inverse_spacing) {
  Value result = 0.0;
  for (std::size_t column = 0; column < point_count; ++column) {
    result += d42_derivative_coefficient(point_count, row, column) *
              values[column];
  }
  return inverse_spacing * result;
}

template <class Value>
void d42_first_derivative(const UniformRadialGrid& grid,
                          const Value* const values,
                          Value* const derivative) {
  if (grid.size() < d42_minimum_points) {
    throw std::invalid_argument("D4-2 SBP requires at least eight points");
  }
  if (values == derivative) {
    throw std::invalid_argument("D4-2 SBP input and output must not alias");
  }
  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t row = 0; row < grid.size(); ++row) {
    derivative[row] = d42_first_derivative_at(
        values, grid.size(), row, inverse_spacing);
  }
}

template <class Value, class InputAllocator, class OutputAllocator>
void d42_first_derivative(
    const UniformRadialGrid& grid,
    const std::vector<Value, InputAllocator>& values,
    std::vector<Value, OutputAllocator>& derivative) {
  if (values.size() != grid.size() || derivative.size() != grid.size()) {
    throw std::invalid_argument("D4-2 SBP buffers must match the grid size");
  }
  d42_first_derivative(grid, values.data(), derivative.data());
}

template <class Value>
struct D42DissipationWorkspace {
  explicit D42DissipationWorkspace(const std::size_t point_count)
      : third_difference(point_count >= 3 ? point_count - 3 : 0),
        normal_product(point_count) {}

  [[nodiscard]] std::size_t size() const { return normal_product.size(); }

  std::vector<Value> third_difference;
  std::vector<Value> normal_product;
};

// Norm-compatible sixth-derivative (KO-like) dissipation reference.
// Let A be the undivided third difference [-1,3,-3,1].  This applies
//
//   Q = -(strength/h) Htilde^{-1} A^T A,    H = h Htilde,
//
// and therefore exactly satisfies u^* H Q u = -strength ||A u||^2 <= 0.
// It annihilates quadratics, is O(h^5) in the interior on smooth fields, and
// retains the D4-2 boundary design order.  The strength includes any desired
// conventional KO normalization (for example 1/64).
template <class Value>
void apply_d42_compatible_dissipation(
    const UniformRadialGrid& grid, const std::vector<Value>& values,
    const double strength, D42DissipationWorkspace<Value>& workspace,
    std::vector<Value>& dissipation) {
  const std::size_t n = grid.size();
  if (n < d42_minimum_points || values.size() != n ||
      workspace.size() != n || dissipation.size() != n) {
    throw std::invalid_argument(
        "D4-2 dissipation buffers must match a grid of at least eight points");
  }
  if (strength < 0.0) {
    throw std::invalid_argument("dissipation strength must be nonnegative");
  }

  for (std::size_t row = 0; row + 3 < n; ++row) {
    workspace.third_difference[row] =
        -values[row] + 3.0 * values[row + 1] -
        3.0 * values[row + 2] + values[row + 3];
  }
  for (std::size_t i = 0; i < n; ++i) workspace.normal_product[i] = 0.0;
  for (std::size_t row = 0; row + 3 < n; ++row) {
    const Value difference = workspace.third_difference[row];
    workspace.normal_product[row] -= difference;
    workspace.normal_product[row + 1] += 3.0 * difference;
    workspace.normal_product[row + 2] -= 3.0 * difference;
    workspace.normal_product[row + 3] += difference;
  }

  for (std::size_t i = 0; i < n; ++i) {
    dissipation[i] =
        -strength * workspace.normal_product[i] /
        (grid.spacing() * d42_norm_weight(n, i));
  }
}

}  // namespace teuk
