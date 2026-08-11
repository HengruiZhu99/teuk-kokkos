#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/grid.hpp"

namespace teuk {

// Readable fourth-order first derivative on a uniform grid.  The two points
// at either end use explicit fourth-order one-sided closures; no boundary
// values are silently imposed.  This is the transparent reference operator,
// retained even when an SBP operator is added for production evolution.
template <class Value>
KOKKOS_INLINE_FUNCTION Value fourth_order_radial_derivative_at(
    const Value* const values, const std::size_t point_count,
    const std::size_t index, const double inverse_spacing) {
  if (index == 0) {
    return inverse_spacing *
           ((-25.0 * values[0] + 48.0 * values[1] - 36.0 * values[2] +
             16.0 * values[3] - 3.0 * values[4]) /
            12.0);
  }
  if (index == 1) {
    return inverse_spacing *
           ((-3.0 * values[0] - 10.0 * values[1] + 18.0 * values[2] -
             6.0 * values[3] + values[4]) /
            12.0);
  }
  if (index + 2 < point_count) {
    return inverse_spacing *
           ((values[index - 2] - 8.0 * values[index - 1] +
             8.0 * values[index + 1] - values[index + 2]) /
            12.0);
  }
  if (index + 1 < point_count) {
    return inverse_spacing *
           ((-values[point_count - 5] + 6.0 * values[point_count - 4] -
             18.0 * values[point_count - 3] +
             10.0 * values[point_count - 2] +
             3.0 * values[point_count - 1]) /
            12.0);
  }
  return inverse_spacing *
         ((3.0 * values[point_count - 5] -
           16.0 * values[point_count - 4] +
           36.0 * values[point_count - 3] -
           48.0 * values[point_count - 2] +
           25.0 * values[point_count - 1]) /
          12.0);
}

template <class Value>
void fourth_order_radial_derivative(const UniformRadialGrid& grid,
                                    const Value* const values,
                                    Value* const derivative) {
  if (values == derivative) {
    throw std::invalid_argument(
        "radial differentiation input and output must not alias");
  }
  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t i = 0; i < grid.size(); ++i) {
    derivative[i] = fourth_order_radial_derivative_at(
        values, grid.size(), i, inverse_spacing);
  }
}

template <class Value, class InputAllocator, class OutputAllocator>
void fourth_order_radial_derivative(
    const UniformRadialGrid& grid,
    const std::vector<Value, InputAllocator>& values,
    std::vector<Value, OutputAllocator>& derivative) {
  if (values.size() != grid.size() || derivative.size() != grid.size()) {
    throw std::invalid_argument(
        "radial differentiation buffers must match the grid size");
  }
  fourth_order_radial_derivative(grid, values.data(), derivative.data());
}

}  // namespace teuk
