#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/grid.hpp"

namespace teuk {

// Classical diagonal-norm D8-4 SBP first derivative: eighth-order centered
// interior rows and fourth-order boundary closures. Coefficients are those of
// Mattsson and Nordstrom, JCP 199 (2004), 503-540, as independently exposed
// by SummationByPartsOperators.jl. They are kept as exact rational literals so
// the SBP identity and polynomial exactness remain executable regressions.
inline constexpr std::size_t d84_minimum_points = 16;
inline constexpr std::size_t d84_boundary_width = 8;

KOKKOS_INLINE_FUNCTION double d84_norm_weight(
    const std::size_t point_count, const std::size_t index) {
  const std::size_t distance =
      index < point_count - 1 - index ? index : point_count - 1 - index;
  if (distance == 0) return 1498139.0 / 5080320.0;
  if (distance == 1) return 1107307.0 / 725760.0;
  if (distance == 2) return 20761.0 / 80640.0;
  if (distance == 3) return 1304999.0 / 725760.0;
  if (distance == 4) return 299527.0 / 725760.0;
  if (distance == 5) return 103097.0 / 80640.0;
  if (distance == 6) return 670091.0 / 725760.0;
  if (distance == 7) return 5127739.0 / 5080320.0;
  return 1.0;
}

KOKKOS_INLINE_FUNCTION double d84_left_boundary_coefficient(
    const std::size_t row, const std::size_t column) {
  if (row == 0) {
    if (column == 0) return -2540160.0 / 1498139.0;
    if (column == 1) return 5544277.0 / 5992556.0;
    if (column == 2) return 198794991.0 / 29962780.0;
    if (column == 3) return -256916579.0 / 17977668.0;
    if (column == 4) return 20708767.0 / 1498139.0;
    if (column == 5) return -41004357.0 / 5992556.0;
    if (column == 6) return 27390659.0 / 17977668.0;
    if (column == 7) return -2323531.0 / 29962780.0;
  } else if (row == 1) {
    if (column == 0) return -5544277.0 / 31004596.0;
    if (column == 1) return 0.0;
    if (column == 2) return -85002381.0 / 22146140.0;
    if (column == 3) return 49607267.0 / 4429228.0;
    if (column == 4) return -165990199.0 / 13287684.0;
    if (column == 5) return 7655859.0 / 1107307.0;
    if (column == 6) return -7568311.0 / 4429228.0;
    if (column == 7) return 48319961.0 / 465068940.0;
  } else if (row == 2) {
    if (column == 0) return -66264997.0 / 8719620.0;
    if (column == 1) return 9444709.0 / 415220.0;
    if (column == 2) return 0.0;
    if (column == 3) return -20335981.0 / 249132.0;
    if (column == 4) return 32320879.0 / 249132.0;
    if (column == 5) return -35518713.0 / 415220.0;
    if (column == 6) return 2502774.0 / 103805.0;
    if (column == 7) return -3177073.0 / 1743924.0;
  } else if (row == 3) {
    if (column == 0) return 256916579.0 / 109619916.0;
    if (column == 1) return -49607267.0 / 5219996.0;
    if (column == 2) return 61007943.0 / 5219996.0;
    if (column == 3) return 0.0;
    if (column == 4) return -68748371.0 / 5219996.0;
    if (column == 5) return 65088123.0 / 5219996.0;
    if (column == 6) return -66558305.0 / 15659988.0;
    if (column == 7) return 3870214.0 / 9134993.0;
  } else if (row == 4) {
    if (column == 0) return -20708767.0 / 2096689.0;
    if (column == 1) return 165990199.0 / 3594324.0;
    if (column == 2) return -96962637.0 / 1198108.0;
    if (column == 3) return 68748371.0 / 1198108.0;
    if (column == 4) return 0.0;
    if (column == 5) return -27294549.0 / 1198108.0;
    if (column == 6) return 14054993.0 / 1198108.0;
    if (column == 7) return -42678199.0 / 25160268.0;
    if (column == 8) return -2592.0 / 299527.0;
  } else if (row == 5) {
    if (column == 0) return 13668119.0 / 8660148.0;
    if (column == 1) return -850651.0 / 103097.0;
    if (column == 2) return 35518713.0 / 2061940.0;
    if (column == 3) return -21696041.0 / 1237164.0;
    if (column == 4) return 9098183.0 / 1237164.0;
    if (column == 5) return 0.0;
    if (column == 6) return -231661.0 / 412388.0;
    if (column == 7) return 7120007.0 / 43300740.0;
    if (column == 8) return 3072.0 / 103097.0;
    if (column == 9) return -288.0 / 103097.0;
  } else if (row == 6) {
    if (column == 0) return -27390659.0 / 56287644.0;
    if (column == 1) return 7568311.0 / 2680364.0;
    if (column == 2) return -22524966.0 / 3350455.0;
    if (column == 3) return 66558305.0 / 8041092.0;
    if (column == 4) return -14054993.0 / 2680364.0;
    if (column == 5) return 2084949.0 / 2680364.0;
    if (column == 6) return 0.0;
    if (column == 7) return 70710683.0 / 93812740.0;
    if (column == 8) return -145152.0 / 670091.0;
    if (column == 9) return 27648.0 / 670091.0;
    if (column == 10) return -2592.0 / 670091.0;
  } else if (row == 7) {
    if (column == 0) return 2323531.0 / 102554780.0;
    if (column == 1) return -48319961.0 / 307664340.0;
    if (column == 2) return 9531219.0 / 20510956.0;
    if (column == 3) return -3870214.0 / 5127739.0;
    if (column == 4) return 2246221.0 / 3238572.0;
    if (column == 5) return -21360021.0 / 102554780.0;
    if (column == 6) return -70710683.0 / 102554780.0;
    if (column == 7) return 0.0;
    if (column == 8) return 4064256.0 / 5127739.0;
    if (column == 9) return -1016064.0 / 5127739.0;
    if (column == 10) return 193536.0 / 5127739.0;
    if (column == 11) return -18144.0 / 5127739.0;
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION double d84_derivative_coefficient(
    const std::size_t point_count, const std::size_t row,
    const std::size_t column) {
  if (row < d84_boundary_width) {
    return d84_left_boundary_coefficient(row, column);
  }
  if (row + d84_boundary_width >= point_count) {
    return -d84_left_boundary_coefficient(point_count - 1 - row,
                                           point_count - 1 - column);
  }
  if (column + 4 == row) return 1.0 / 280.0;
  if (column + 3 == row) return -4.0 / 105.0;
  if (column + 2 == row) return 1.0 / 5.0;
  if (column + 1 == row) return -4.0 / 5.0;
  if (column == row + 1) return 4.0 / 5.0;
  if (column == row + 2) return -1.0 / 5.0;
  if (column == row + 3) return 4.0 / 105.0;
  if (column == row + 4) return -1.0 / 280.0;
  return 0.0;
}

KOKKOS_INLINE_FUNCTION double d84_norm_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return row == column ? grid.spacing() * d84_norm_weight(grid.size(), row)
                       : 0.0;
}

KOKKOS_INLINE_FUNCTION double d84_derivative_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return d84_derivative_coefficient(grid.size(), row, column) /
         grid.spacing();
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value d84_first_derivative_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t row, const double inverse_spacing) {
  Value result = 0.0;
  if (row < d84_boundary_width) {
    for (std::size_t column = 0; column < 12; ++column) {
      result += d84_left_boundary_coefficient(row, column) * values[column];
    }
  } else if (row + d84_boundary_width >= point_count) {
    const std::size_t reflected_row = point_count - 1 - row;
    for (std::size_t reflected_column = 0; reflected_column < 12;
         ++reflected_column) {
      result -= d84_left_boundary_coefficient(reflected_row,
                                              reflected_column) *
                values[point_count - 1 - reflected_column];
    }
  } else {
    result = (values[row - 4] - values[row + 4]) / 280.0 +
             4.0 * (-values[row - 3] + values[row + 3]) / 105.0 +
             (values[row - 2] - values[row + 2]) / 5.0 +
             4.0 * (-values[row - 1] + values[row + 1]) / 5.0;
  }
  return inverse_spacing * result;
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value d84_first_derivative_strided_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t row, const double inverse_spacing,
    const std::size_t stride) {
  Value result = 0.0;
  if (row < d84_boundary_width) {
    for (std::size_t column = 0; column < 12; ++column) {
      result += d84_left_boundary_coefficient(row, column) *
                values[column * stride];
    }
  } else if (row + d84_boundary_width >= point_count) {
    const std::size_t reflected_row = point_count - 1 - row;
    for (std::size_t reflected_column = 0; reflected_column < 12;
         ++reflected_column) {
      result -= d84_left_boundary_coefficient(reflected_row,
                                              reflected_column) *
                values[(point_count - 1 - reflected_column) * stride];
    }
  } else {
    result = (values[(row - 4) * stride] - values[(row + 4) * stride]) /
                 280.0 +
             4.0 * (-values[(row - 3) * stride] +
                    values[(row + 3) * stride]) /
                 105.0 +
             (values[(row - 2) * stride] - values[(row + 2) * stride]) /
                 5.0 +
             4.0 * (-values[(row - 1) * stride] +
                    values[(row + 1) * stride]) /
                 5.0;
  }
  return inverse_spacing * result;
}

template <class Value>
void d84_first_derivative(const UniformRadialGrid& grid,
                          const Value* const values,
                          Value* const derivative) {
  if (grid.size() < d84_minimum_points) {
    throw std::invalid_argument("D8-4 SBP requires at least sixteen points");
  }
  if (values == derivative) {
    throw std::invalid_argument("D8-4 SBP input and output must not alias");
  }
  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t row = 0; row < grid.size(); ++row) {
    derivative[row] = d84_first_derivative_at(
        values, grid.size(), row, inverse_spacing);
  }
}

template <class Value, class InputAllocator, class OutputAllocator>
void d84_first_derivative(
    const UniformRadialGrid& grid,
    const std::vector<Value, InputAllocator>& values,
    std::vector<Value, OutputAllocator>& derivative) {
  if (values.size() != grid.size() || derivative.size() != grid.size()) {
    throw std::invalid_argument("D8-4 SBP buffers must match the grid size");
  }
  d84_first_derivative(grid, values.data(), derivative.data());
}

// Energy-compatible tenth-derivative KO-like dissipation. A is the undivided
// fifth difference [-1,5,-10,10,-5,1] and
//
//   Q = -(strength/h) Htilde^{-1} A^T A.
//
// Thus u* H Q u=-strength ||A u||^2 <= 0 exactly. It annihilates quartics,
// is O(h^9) in the interior, and O(h^4) at the boundary, preserving D8-4's
// fourth-order boundary design.
template <class Value>
KOKKOS_INLINE_FUNCTION Value d84_compatible_dissipation_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t index, const double spacing, const double strength) {
  const std::size_t first_row = index > 5 ? index - 5 : 0;
  const std::size_t last_row =
      index < point_count - 5 ? index : point_count - 6;
  Value normal_product = 0.0;
  for (std::size_t row = first_row; row <= last_row; ++row) {
    const Value difference =
        -values[row] + 5.0 * values[row + 1] -
        10.0 * values[row + 2] + 10.0 * values[row + 3] -
        5.0 * values[row + 4] + values[row + 5];
    const std::size_t position = index - row;
    const double transpose =
        position == 0   ? -1.0
        : position == 1 ? 5.0
        : position == 2 ? -10.0
        : position == 3 ? 10.0
        : position == 4 ? -5.0
                        : 1.0;
    normal_product += transpose * difference;
  }
  return -strength * normal_product /
         (spacing * d84_norm_weight(point_count, index));
}

}  // namespace teuk
