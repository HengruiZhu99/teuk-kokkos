#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/sbp.hpp"
#include "teuk/sbp84.hpp"

namespace teuk {

enum class RadialDiscretization { D42, D84 };

inline const char* radial_discretization_name(
    const RadialDiscretization discretization) {
  switch (discretization) {
    case RadialDiscretization::D42:
      return "d4-2";
    case RadialDiscretization::D84:
      return "d8-4";
  }
  throw std::invalid_argument("unknown radial discretization");
}

inline RadialDiscretization parse_radial_discretization(
    const std::string& text) {
  if (text == "d4-2") return RadialDiscretization::D42;
  if (text == "d8-4") return RadialDiscretization::D84;
  throw std::invalid_argument("unknown radial discretization: " + text);
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t radial_minimum_points(
    const RadialDiscretization discretization) {
  return discretization == RadialDiscretization::D84 ? d84_minimum_points
                                                      : d42_minimum_points;
}

KOKKOS_INLINE_FUNCTION double radial_norm_weight(
    const RadialDiscretization discretization,
    const std::size_t point_count, const std::size_t index) {
  return discretization == RadialDiscretization::D84
             ? d84_norm_weight(point_count, index)
             : d42_norm_weight(point_count, index);
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value radial_first_derivative_at(
    const RadialDiscretization discretization, const Value* const values,
    const std::size_t point_count, const std::size_t row,
    const double inverse_spacing) {
  return discretization == RadialDiscretization::D84
             ? d84_first_derivative_at(values, point_count, row,
                                       inverse_spacing)
             : d42_first_derivative_at(values, point_count, row,
                                       inverse_spacing);
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value radial_first_derivative_strided_at(
    const RadialDiscretization discretization, const Value* const values,
    const std::size_t point_count, const std::size_t row,
    const double inverse_spacing, const std::size_t stride) {
  return discretization == RadialDiscretization::D84
             ? d84_first_derivative_strided_at(
                   values, point_count, row, inverse_spacing, stride)
             : d42_first_derivative_strided_at(
                   values, point_count, row, inverse_spacing, stride);
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value radial_compatible_dissipation_at(
    const RadialDiscretization discretization, const Value* const values,
    const std::size_t point_count, const std::size_t index,
    const double spacing, const double strength) {
  return discretization == RadialDiscretization::D84
             ? d84_compatible_dissipation_at(values, point_count, index,
                                             spacing, strength)
             : d42_compatible_dissipation_at(values, point_count, index,
                                             spacing, strength);
}

// Strided counterpart for LayoutRight (mode,field,radial,theta) storage.
// The algebra is the same negative-semidefinite A^T A form as the contiguous
// operators above; only the radial memory stride differs.
template <class Value>
KOKKOS_INLINE_FUNCTION Value radial_compatible_dissipation_at(
    const RadialDiscretization discretization, const Value* const values,
    const std::size_t point_count, const std::size_t index,
    const double spacing, const double strength, const std::size_t stride) {
  const std::size_t difference_order =
      discretization == RadialDiscretization::D84 ? 5 : 3;
  const std::size_t first_row =
      index > difference_order ? index - difference_order : 0;
  const std::size_t last_row =
      index < point_count - difference_order
          ? index
          : point_count - difference_order - 1;
  Value normal_product = 0.0;
  for (std::size_t row = first_row; row <= last_row; ++row) {
    Value difference = 0.0;
    if (discretization == RadialDiscretization::D84) {
      difference = -values[row * stride] +
                   5.0 * values[(row + 1) * stride] -
                   10.0 * values[(row + 2) * stride] +
                   10.0 * values[(row + 3) * stride] -
                   5.0 * values[(row + 4) * stride] +
                   values[(row + 5) * stride];
    } else {
      difference = -values[row * stride] +
                   3.0 * values[(row + 1) * stride] -
                   3.0 * values[(row + 2) * stride] +
                   values[(row + 3) * stride];
    }
    const std::size_t position = index - row;
    double transpose = 0.0;
    if (discretization == RadialDiscretization::D84) {
      transpose = position == 0   ? -1.0
                  : position == 1 ? 5.0
                  : position == 2 ? -10.0
                  : position == 3 ? 10.0
                  : position == 4 ? -5.0
                                  : 1.0;
    } else {
      transpose = position == 0
                      ? -1.0
                      : (position == 1
                             ? 3.0
                             : (position == 2 ? -3.0 : 1.0));
    }
    normal_product += transpose * difference;
  }
  return -strength * normal_product /
         (spacing * radial_norm_weight(discretization, point_count, index));
}

}  // namespace teuk
