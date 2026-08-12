#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "teuk/sbp.hpp"
#include "teuk/sbp105.hpp"
#include "teuk/sbp84.hpp"

namespace teuk {

enum class RadialDiscretization { D42, D84, D105 };

// Conservative integer upper bounds for rho(Htilde^-1 A^T A). The nonzero
// eigenvalues equal those of the smaller symmetric PSD matrix
// A Htilde^-1 A^T, whose finite boundary/interior row patterns give maximum
// absolute row sums 64, 1370.20793, and 5497.33017 respectively.
inline constexpr double d42_dissipation_spectral_radius_bound = 64.0;
inline constexpr double d84_dissipation_spectral_radius_bound = 1371.0;
inline constexpr double d105_dissipation_spectral_radius_bound = 5500.0;
inline constexpr double rk4_negative_real_axis_extent =
    2.785293563405282;
inline constexpr double radial_dissipation_rk4_safety_factor = 0.8;

inline double radial_dissipation_spectral_radius_bound(
    const RadialDiscretization discretization) {
  switch (discretization) {
    case RadialDiscretization::D42:
      return d42_dissipation_spectral_radius_bound;
    case RadialDiscretization::D84:
      return d84_dissipation_spectral_radius_bound;
    case RadialDiscretization::D105:
      return d105_dissipation_spectral_radius_bound;
  }
  throw std::invalid_argument("unknown radial discretization");
}

inline double radial_dissipation_rk4_maximum_step_ratio(
    const RadialDiscretization discretization, const double strength) {
  if (!std::isfinite(strength) || strength < 0.0) {
    throw std::invalid_argument(
        "dissipation strength must be finite and nonnegative");
  }
  if (strength == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return radial_dissipation_rk4_safety_factor *
         rk4_negative_real_axis_extent /
         (strength *
          radial_dissipation_spectral_radius_bound(discretization));
}

inline bool radial_dissipation_rk4_step_is_admissible(
    const RadialDiscretization discretization, const double strength,
    const double step_over_spacing) {
  if (!std::isfinite(strength) || strength < 0.0 ||
      !std::isfinite(step_over_spacing) || step_over_spacing < 0.0) {
    return false;
  }
  return step_over_spacing <=
         radial_dissipation_rk4_maximum_step_ratio(discretization, strength);
}

inline const char* radial_discretization_name(
    const RadialDiscretization discretization) {
  switch (discretization) {
    case RadialDiscretization::D42:
      return "d4-2";
    case RadialDiscretization::D84:
      return "d8-4";
    case RadialDiscretization::D105:
      return "d10-5";
  }
  throw std::invalid_argument("unknown radial discretization");
}

inline RadialDiscretization parse_radial_discretization(
    const std::string& text) {
  if (text == "d4-2") return RadialDiscretization::D42;
  if (text == "d8-4") return RadialDiscretization::D84;
  if (text == "d10-5") return RadialDiscretization::D105;
  throw std::invalid_argument("unknown radial discretization: " + text);
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t radial_minimum_points(
    const RadialDiscretization discretization) {
  switch (discretization) {
    case RadialDiscretization::D42:
      return d42_minimum_points;
    case RadialDiscretization::D84:
      return d84_minimum_points;
    case RadialDiscretization::D105:
      return d105_minimum_points;
  }
  return 0;
}

KOKKOS_INLINE_FUNCTION double radial_norm_weight(
    const RadialDiscretization discretization,
    const std::size_t point_count, const std::size_t index) {
  if (discretization == RadialDiscretization::D105) {
    return d105_norm_weight(point_count, index);
  }
  return discretization == RadialDiscretization::D84
             ? d84_norm_weight(point_count, index)
             : d42_norm_weight(point_count, index);
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value radial_first_derivative_at(
    const RadialDiscretization discretization, const Value* const values,
    const std::size_t point_count, const std::size_t row,
    const double inverse_spacing) {
  if (discretization == RadialDiscretization::D105) {
    return d105_first_derivative_at(values, point_count, row,
                                    inverse_spacing);
  }
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
  if (discretization == RadialDiscretization::D105) {
    return d105_first_derivative_strided_at(
        values, point_count, row, inverse_spacing, stride);
  }
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
  if (discretization == RadialDiscretization::D105) {
    return d105_compatible_dissipation_at(values, point_count, index, spacing,
                                          strength);
  }
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
      discretization == RadialDiscretization::D105
          ? 6
          : (discretization == RadialDiscretization::D84 ? 5 : 3);
  const std::size_t first_row =
      index > difference_order ? index - difference_order : 0;
  const std::size_t last_row =
      index < point_count - difference_order
          ? index
          : point_count - difference_order - 1;
  Value normal_product = 0.0;
  for (std::size_t row = first_row; row <= last_row; ++row) {
    Value difference = 0.0;
    if (discretization == RadialDiscretization::D105) {
      difference = values[row * stride] -
                   6.0 * values[(row + 1) * stride] +
                   15.0 * values[(row + 2) * stride] -
                   20.0 * values[(row + 3) * stride] +
                   15.0 * values[(row + 4) * stride] -
                   6.0 * values[(row + 5) * stride] +
                   values[(row + 6) * stride];
    } else if (discretization == RadialDiscretization::D84) {
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
    if (discretization == RadialDiscretization::D105) {
      transpose = position == 0   ? 1.0
                  : position == 1 ? -6.0
                  : position == 2 ? 15.0
                  : position == 3 ? -20.0
                  : position == 4 ? 15.0
                  : position == 5 ? -6.0
                                  : 1.0;
    } else if (discretization == RadialDiscretization::D84) {
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
