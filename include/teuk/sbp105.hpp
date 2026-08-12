#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/grid.hpp"

namespace teuk {

// Diagonal-norm D10-5 SBP first derivative: tenth-order centered interior
// rows and fifth-order boundary closures. This is the exact-rational,
// uniform-grid operator distributed as d2_10.m by the MIT-licensed sbplib
// reference implementation (scicompuu/sbplib, commit e64d8c6). It is a
// practical low-spectral-radius closure, but is distinct from the rounded
// minimum-ABTE D10-5 operator tabulated by Diener et al., JSC 32 (2007).
//
// Eleven boundary rows and a maximum 16-point left stencil require at least
// 22 grid points so the two closure regions do not overlap.
inline constexpr std::size_t d105_minimum_points = 22;
inline constexpr std::size_t d105_boundary_width = 11;
inline constexpr std::size_t d105_boundary_stencil_width = 16;

KOKKOS_INLINE_FUNCTION double d105_norm_weight(
    const std::size_t point_count, const std::size_t index) {
  const std::size_t distance =
      index < point_count - 1 - index ? index : point_count - 1 - index;
  if (distance == 0) return 5261271563.0 / 18289152000.0;
  if (distance == 1) return 2881040311.0 / 1828915200.0;
  if (distance == 2) return 52175551.0 / 406425600.0;
  if (distance == 3) return 11662993.0 / 6096384.0;
  if (distance == 4) return 50124587.0 / 87091200.0;
  if (distance == 5) return 50124587.0 / 72576000.0;
  if (distance == 6) return 148333439.0 / 87091200.0;
  if (distance == 7) return 63867949.0 / 152409600.0;
  if (distance == 8) return 20608675.0 / 16257024.0;
  if (distance == 9) return 1704508063.0 / 1828915200.0;
  if (distance == 10) return 18425967263.0 / 18289152000.0;
  return 1.0;
}

KOKKOS_INLINE_FUNCTION double d105_interior_coefficient(
    const std::size_t row, const std::size_t column) {
  if (column + 5 == row) return -1.0 / 1260.0;
  if (column + 4 == row) return 5.0 / 504.0;
  if (column + 3 == row) return -5.0 / 84.0;
  if (column + 2 == row) return 5.0 / 21.0;
  if (column + 1 == row) return -5.0 / 6.0;
  if (column == row + 1) return 5.0 / 6.0;
  if (column == row + 2) return -5.0 / 21.0;
  if (column == row + 3) return 5.0 / 84.0;
  if (column == row + 4) return -5.0 / 504.0;
  if (column == row + 5) return 1.0 / 1260.0;
  return 0.0;
}

// The left 11-by-11 block of Q=Htilde*D. Q is skew-symmetric except for
// Q(0,0)=-1/2; entries coupling a closure row to columns 11..15 follow from
// skew symmetry and the centered interior stencil.
KOKKOS_INLINE_FUNCTION double d105_left_q_coefficient(
    const std::size_t row, const std::size_t column) {
  if (column >= d105_boundary_width) {
    return d105_interior_coefficient(row, column);
  }
  if (row == 0) {
    if (column == 0) return -1.0 / 2.0;
    if (column == 1) return 2300876759589119.0 / 3395198177280000.0;
    if (column == 2) return -99808615498093.0 / 2263465451520000.0;
    if (column == 3) return -34957747037683.0 / 212199886080000.0;
    if (column == 4) return -709586095717.0 / 13473008640000.0;
    if (column == 5) return 325330433051.0 / 6218311680000.0;
    if (column == 6) return 27953548723573.0 / 485028311040000.0;
    if (column == 7) return 2690678501.0 / 412439040000.0;
    if (column == 8) return -2397491025029.0 / 70733295360000.0;
    if (column == 9) return -9959492094287.0 / 1131732725760000.0;
    if (column == 10) return 5242772857661.0 / 522338181120000.0;
  } else if (row == 1) {
    if (column == 0) return -2300876759589119.0 / 3395198177280000.0;
    if (column == 1) return 0.0;
    if (column == 2) return 3103439505511.0 / 16643128320000.0;
    if (column == 3) return 2700334568377.0 / 5052378240000.0;
    if (column == 4) return 50587599589937.0 / 242514155520000.0;
    if (column == 5) return -5570893587157.0 / 40419025920000.0;
    if (column == 6) return -1496329934863.0 / 8083805184000.0;
    if (column == 7) return -322512443237.0 / 12482346240000.0;
    if (column == 8) return 2275340833763.0 / 23096586240000.0;
    if (column == 9) return 22922115021893.0 / 848799544320000.0;
    if (column == 10) return -143.0 / 5000.0;
  } else if (row == 2) {
    if (column == 0) return 99808615498093.0 / 2263465451520000.0;
    if (column == 1) return -3103439505511.0 / 16643128320000.0;
    if (column == 2) return 0.0;
    if (column == 3) return 15053664233879.0 / 40419025920000.0;
    if (column == 4) return -9306441440587.0 / 32335220736000.0;
    if (column == 5) return -945459729233.0 / 13473008640000.0;
    if (column == 6) return 956829267413.0 / 5774146560000.0;
    if (column == 7) return 446866085903.0 / 56586636288000.0;
    if (column == 8) return -41109372242993.0 / 754488483840000.0;
    if (column == 9) return 1.0 / 500.0;
    if (column == 10) return 17.0 / 2500.0;
  } else if (row == 3) {
    if (column == 0) return 34957747037683.0 / 212199886080000.0;
    if (column == 1) return -2700334568377.0 / 5052378240000.0;
    if (column == 2) return -15053664233879.0 / 40419025920000.0;
    if (column == 3) return 0.0;
    if (column == 4) return 3899174751943.0 / 10104756480000.0;
    if (column == 5) return 4691717443831.0 / 10104756480000.0;
    if (column == 6) return -58571891887.0 / 396264960000.0;
    if (column == 7) return 100791910589.0 / 1040195520000.0;
    if (column == 8) return -425149181.0 / 29719872000.0;
    if (column == 9) return -2376515922259.0 / 30314269440000.0;
    if (column == 10) return 36894656431.0 / 1036385280000.0;
  } else if (row == 4) {
    if (column == 0) return 709586095717.0 / 13473008640000.0;
    if (column == 1) return -50587599589937.0 / 242514155520000.0;
    if (column == 2) return 9306441440587.0 / 32335220736000.0;
    if (column == 3) return -3899174751943.0 / 10104756480000.0;
    if (column == 4) return 0.0;
    if (column == 5) return -4552305973.0 / 444165120000.0;
    if (column == 6) return 4984940784247.0 / 11548293120000.0;
    if (column == 7) return -19410791.0 / 146764800.0;
    if (column == 8) return -2912773695913.0 / 40419025920000.0;
    if (column == 9) return 127067639161.0 / 3233522073600.0;
    if (column == 10) return -89277540287.0 / 37309870080000.0;
  } else if (row == 5) {
    if (column == 0) return -325330433051.0 / 6218311680000.0;
    if (column == 1) return 5570893587157.0 / 40419025920000.0;
    if (column == 2) return 945459729233.0 / 13473008640000.0;
    if (column == 3) return -4691717443831.0 / 10104756480000.0;
    if (column == 4) return 4552305973.0 / 444165120000.0;
    if (column == 5) return 0.0;
    if (column == 6) return 31722122083.0 / 84913920000.0;
    if (column == 7) return -887187251021.0 / 10104756480000.0;
    if (column == 8) return -1661755478749.0 / 26946017280000.0;
    if (column == 9) return 1505713246249.0 / 13473008640000.0;
    if (column == 10) return -38859042469.0 / 1036385280000.0;
  } else if (row == 6) {
    if (column == 0) return -27953548723573.0 / 485028311040000.0;
    if (column == 1) return 1496329934863.0 / 8083805184000.0;
    if (column == 2) return -956829267413.0 / 5774146560000.0;
    if (column == 3) return 58571891887.0 / 396264960000.0;
    if (column == 4) return -4984940784247.0 / 11548293120000.0;
    if (column == 5) return -31722122083.0 / 84913920000.0;
    if (column == 6) return 0.0;
    if (column == 7) return 9357094407023.0 / 20209512960000.0;
    if (column == 8) return 52602356173249.0 / 161676103680000.0;
    if (column == 9) return -1435252677707.0 / 17322439680000.0;
    if (column == 10) return -33048158431.0 / 3109155840000.0;
  } else if (row == 7) {
    if (column == 0) return -2690678501.0 / 412439040000.0;
    if (column == 1) return 322512443237.0 / 12482346240000.0;
    if (column == 2) return -446866085903.0 / 56586636288000.0;
    if (column == 3) return -100791910589.0 / 1040195520000.0;
    if (column == 4) return 19410791.0 / 146764800.0;
    if (column == 5) return 887187251021.0 / 10104756480000.0;
    if (column == 6) return -9357094407023.0 / 20209512960000.0;
    if (column == 7) return 0.0;
    if (column == 8) return 70089734285659.0 / 141466590720000.0;
    if (column == 9) return -105938137621.0 / 471555302400.0;
    if (column == 10) return 4358988450443.0 / 65292272640000.0;
  } else if (row == 8) {
    if (column == 0) return 2397491025029.0 / 70733295360000.0;
    if (column == 1) return -2275340833763.0 / 23096586240000.0;
    if (column == 2) return 41109372242993.0 / 754488483840000.0;
    if (column == 3) return 425149181.0 / 29719872000.0;
    if (column == 4) return 2912773695913.0 / 40419025920000.0;
    if (column == 5) return 1661755478749.0 / 26946017280000.0;
    if (column == 6) return -52602356173249.0 / 161676103680000.0;
    if (column == 7) return -70089734285659.0 / 141466590720000.0;
    if (column == 8) return 0.0;
    if (column == 9) return 314274398580227.0 / 377244241920000.0;
    if (column == 10) return -97822819709.0 / 487710720000.0;
  } else if (row == 9) {
    if (column == 0) return 9959492094287.0 / 1131732725760000.0;
    if (column == 1) return -22922115021893.0 / 848799544320000.0;
    if (column == 2) return -1.0 / 500.0;
    if (column == 3) return 2376515922259.0 / 30314269440000.0;
    if (column == 4) return -127067639161.0 / 3233522073600.0;
    if (column == 5) return -1505713246249.0 / 13473008640000.0;
    if (column == 6) return 1435252677707.0 / 17322439680000.0;
    if (column == 7) return 105938137621.0 / 471555302400.0;
    if (column == 8) return -314274398580227.0 / 377244241920000.0;
    if (column == 9) return 0.0;
    if (column == 10) return 7519148725913.0 / 9327467520000.0;
  } else if (row == 10) {
    if (column == 0) return -5242772857661.0 / 522338181120000.0;
    if (column == 1) return 143.0 / 5000.0;
    if (column == 2) return -17.0 / 2500.0;
    if (column == 3) return -36894656431.0 / 1036385280000.0;
    if (column == 4) return 89277540287.0 / 37309870080000.0;
    if (column == 5) return 38859042469.0 / 1036385280000.0;
    if (column == 6) return 33048158431.0 / 3109155840000.0;
    if (column == 7) return -4358988450443.0 / 65292272640000.0;
    if (column == 8) return 97822819709.0 / 487710720000.0;
    if (column == 9) return -7519148725913.0 / 9327467520000.0;
    if (column == 10) return 0.0;
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION double d105_left_boundary_coefficient(
    const std::size_t row, const std::size_t column) {
  return d105_left_q_coefficient(row, column) /
         d105_norm_weight(2 * d105_boundary_width, row);
}

KOKKOS_INLINE_FUNCTION double d105_derivative_coefficient(
    const std::size_t point_count, const std::size_t row,
    const std::size_t column) {
  if (row < d105_boundary_width) {
    return column < d105_boundary_stencil_width
               ? d105_left_boundary_coefficient(row, column)
               : 0.0;
  }
  if (row + d105_boundary_width >= point_count) {
    return -d105_derivative_coefficient(
        point_count, point_count - 1 - row, point_count - 1 - column);
  }
  return d105_interior_coefficient(row, column);
}

KOKKOS_INLINE_FUNCTION double d105_norm_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return row == column ? grid.spacing() * d105_norm_weight(grid.size(), row)
                       : 0.0;
}

KOKKOS_INLINE_FUNCTION double d105_derivative_matrix_entry(
    const UniformRadialGrid& grid, const std::size_t row,
    const std::size_t column) {
  return d105_derivative_coefficient(grid.size(), row, column) /
         grid.spacing();
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value d105_first_derivative_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t row, const double inverse_spacing) {
  Value result = 0.0;
  if (row < d105_boundary_width) {
    for (std::size_t column = 0; column < d105_boundary_stencil_width;
         ++column) {
      result += d105_left_boundary_coefficient(row, column) * values[column];
    }
  } else if (row + d105_boundary_width >= point_count) {
    const std::size_t reflected_row = point_count - 1 - row;
    for (std::size_t reflected_column = 0;
         reflected_column < d105_boundary_stencil_width;
         ++reflected_column) {
      result -= d105_left_boundary_coefficient(reflected_row,
                                               reflected_column) *
                values[point_count - 1 - reflected_column];
    }
  } else {
    result = (-values[row - 5] + values[row + 5]) / 1260.0 +
             5.0 * (values[row - 4] - values[row + 4]) / 504.0 +
             5.0 * (-values[row - 3] + values[row + 3]) / 84.0 +
             5.0 * (values[row - 2] - values[row + 2]) / 21.0 +
             5.0 * (-values[row - 1] + values[row + 1]) / 6.0;
  }
  return inverse_spacing * result;
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value d105_first_derivative_strided_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t row, const double inverse_spacing,
    const std::size_t stride) {
  Value result = 0.0;
  if (row < d105_boundary_width) {
    for (std::size_t column = 0; column < d105_boundary_stencil_width;
         ++column) {
      result += d105_left_boundary_coefficient(row, column) *
                values[column * stride];
    }
  } else if (row + d105_boundary_width >= point_count) {
    const std::size_t reflected_row = point_count - 1 - row;
    for (std::size_t reflected_column = 0;
         reflected_column < d105_boundary_stencil_width;
         ++reflected_column) {
      result -= d105_left_boundary_coefficient(reflected_row,
                                               reflected_column) *
                values[(point_count - 1 - reflected_column) * stride];
    }
  } else {
    result = (-values[(row - 5) * stride] + values[(row + 5) * stride]) /
                 1260.0 +
             5.0 * (values[(row - 4) * stride] -
                    values[(row + 4) * stride]) /
                 504.0 +
             5.0 * (-values[(row - 3) * stride] +
                    values[(row + 3) * stride]) /
                 84.0 +
             5.0 * (values[(row - 2) * stride] -
                    values[(row + 2) * stride]) /
                 21.0 +
             5.0 * (-values[(row - 1) * stride] +
                    values[(row + 1) * stride]) /
                 6.0;
  }
  return inverse_spacing * result;
}

template <class Value>
void d105_first_derivative(const UniformRadialGrid& grid,
                           const Value* const values,
                           Value* const derivative) {
  if (grid.size() < d105_minimum_points) {
    throw std::invalid_argument("D10-5 SBP requires at least twenty-two points");
  }
  if (values == derivative) {
    throw std::invalid_argument("D10-5 SBP input and output must not alias");
  }
  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t row = 0; row < grid.size(); ++row) {
    derivative[row] = d105_first_derivative_at(
        values, grid.size(), row, inverse_spacing);
  }
}

template <class Value, class InputAllocator, class OutputAllocator>
void d105_first_derivative(
    const UniformRadialGrid& grid,
    const std::vector<Value, InputAllocator>& values,
    std::vector<Value, OutputAllocator>& derivative) {
  if (values.size() != grid.size() || derivative.size() != grid.size()) {
    throw std::invalid_argument("D10-5 SBP buffers must match the grid size");
  }
  d105_first_derivative(grid, values.data(), derivative.data());
}

// Energy-compatible twelfth-derivative KO-like dissipation. A is the
// undivided sixth difference [1,-6,15,-20,15,-6,1] and
//
//   Q = -(strength/h) Htilde^{-1} A^T A.
//
// Therefore u* H Q u = -strength ||A u||^2 <= 0. It annihilates
// quintics, is O(h^11) in the interior, and O(h^5) at the boundary.
template <class Value>
KOKKOS_INLINE_FUNCTION Value d105_compatible_dissipation_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t index, const double spacing, const double strength) {
  const std::size_t first_row = index > 6 ? index - 6 : 0;
  const std::size_t last_row = index < point_count - 6 ? index
                                                       : point_count - 7;
  Value normal_product = 0.0;
  for (std::size_t row = first_row; row <= last_row; ++row) {
    const Value difference =
        values[row] - 6.0 * values[row + 1] +
        15.0 * values[row + 2] - 20.0 * values[row + 3] +
        15.0 * values[row + 4] - 6.0 * values[row + 5] + values[row + 6];
    const std::size_t position = index - row;
    const double transpose =
        position == 0   ? 1.0
        : position == 1 ? -6.0
        : position == 2 ? 15.0
        : position == 3 ? -20.0
        : position == 4 ? 15.0
        : position == 5 ? -6.0
                        : 1.0;
    normal_product += transpose * difference;
  }
  return -strength * normal_product /
         (spacing * d105_norm_weight(point_count, index));
}

}  // namespace teuk
