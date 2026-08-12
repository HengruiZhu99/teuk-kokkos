#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "teuk/plus2_endpoint_extraction_weights.hpp"
#include "teuk/routeb_fornberg.hpp"

namespace teuk {

// Extract [R^2]f0 and [R]f1 from strictly positive uniform nodes.  Peeling is
// not assumed or silently imposed: callers must independently inspect f0(0),
// f0'(0), and f1(0), for example with plus2_peeling_residuals_at_scri below.
template <class Value>
KOKKOS_INLINE_FUNCTION Value plus2_extract_q0_at_scri(
    const Value* f0, const std::size_t point_count,
    const double inverse_spacing, const std::size_t stride = 1) {
  if (point_count < plus2_q0_endpoint_nodes + 1 ||
      !(inverse_spacing > 0.0)) {
    return Value{};
  }
  Value result{};
  for (std::size_t node = 0; node < plus2_q0_endpoint_nodes; ++node) {
    result += plus2_q0_endpoint_weights[node] * f0[(node + 1) * stride];
  }
  return inverse_spacing * inverse_spacing * result;
}

template <class Value>
KOKKOS_INLINE_FUNCTION Value plus2_extract_q1_at_scri(
    const Value* f1, const std::size_t point_count,
    const double inverse_spacing, const std::size_t stride = 1) {
  if (point_count < plus2_q1_endpoint_nodes + 1 ||
      !(inverse_spacing > 0.0)) {
    return Value{};
  }
  Value result{};
  for (std::size_t node = 0; node < plus2_q1_endpoint_nodes; ++node) {
    result += plus2_q1_endpoint_weights[node] * f1[(node + 1) * stride];
  }
  return inverse_spacing * result;
}

template <class Value>
struct Plus2PeelingResidualsAtScri {
  Value f0_constant;
  Value f0_linear;
  Value f1_constant;
};

template <class Value>
KOKKOS_INLINE_FUNCTION Plus2PeelingResidualsAtScri<Value>
plus2_peeling_residuals_at_scri(const Value* f0, const Value* f1,
                                const std::size_t point_count,
                                const double inverse_spacing,
                                const std::size_t stride = 1) {
  return {f0[0],
          routeb_fornberg_direct_derivative_at(
              1, f0, point_count, 0, inverse_spacing, stride),
          f1[0]};
}

}  // namespace teuk
