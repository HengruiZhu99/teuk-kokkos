#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/coupled.hpp"
#include "teuk/io.hpp"

namespace teuk {

inline constexpr std::size_t point_pipeline_field_count = 13;

KOKKOS_INLINE_FUNCTION Complex point_pipeline_component(
    const PointPipelineState& state, const std::size_t field) {
  switch (field) {
    case 0: return state.first.P;
    case 1: return state.first.Q;
    case 2: return state.first.psi;
    case 3: return state.reconstruction.G;
    case 4: return state.reconstruction.Lambda;
    case 5: return state.reconstruction.H;
    case 6: return state.reconstruction.B;
    case 7: return state.reconstruction.Pi;
    case 8: return state.reconstruction.C;
    case 9: return state.reconstruction.U;
    case 10: return state.second.P;
    case 11: return state.second.Q;
    case 12: return state.second.psi;
    default: return Complex(0.0, 0.0);
  }
}

KOKKOS_INLINE_FUNCTION void set_point_pipeline_component(
    PointPipelineState& state, const std::size_t field, const Complex value) {
  switch (field) {
    case 0: state.first.P = value; break;
    case 1: state.first.Q = value; break;
    case 2: state.first.psi = value; break;
    case 3: state.reconstruction.G = value; break;
    case 4: state.reconstruction.Lambda = value; break;
    case 5: state.reconstruction.H = value; break;
    case 6: state.reconstruction.B = value; break;
    case 7: state.reconstruction.Pi = value; break;
    case 8: state.reconstruction.C = value; break;
    case 9: state.reconstruction.U = value; break;
    case 10: state.second.P = value; break;
    case 11: state.second.Q = value; break;
    case 12: state.second.psi = value; break;
    default: break;
  }
}

inline std::size_t checked_snapshot_value_count(const SnapshotShape shape) {
  std::size_t count = 1;
  const std::uint64_t extents[] = {shape.modes, shape.fields, shape.radial,
                                   shape.theta};
  for (const std::uint64_t extent : extents) {
    if (extent == 0 ||
        extent > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max() / count)) {
      throw std::invalid_argument("invalid or overflowing snapshot shape");
    }
    count *= static_cast<std::size_t>(extent);
  }
  return count;
}

// Input points use logical (mode,radial,theta) order with theta contiguous.
// Snapshots use the documented (mode,field,radial,theta) order.
inline std::vector<Complex> pack_point_pipeline_snapshot(
    const std::vector<PointPipelineState>& points,
    const SnapshotShape shape) {
  if (shape.fields != point_pipeline_field_count) {
    throw std::invalid_argument("coupled snapshot must contain 13 fields");
  }
  const std::size_t value_count = checked_snapshot_value_count(shape);
  const std::size_t point_count = value_count / point_pipeline_field_count;
  if (points.size() != point_count) {
    throw std::invalid_argument("coupled point count does not match shape");
  }
  std::vector<Complex> values(value_count);
  const std::size_t radial_count = static_cast<std::size_t>(shape.radial);
  const std::size_t theta_count = static_cast<std::size_t>(shape.theta);
  for (std::size_t mode = 0; mode < shape.modes; ++mode) {
    for (std::size_t field = 0; field < shape.fields; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          const std::size_t point =
              (mode * radial_count + radial) * theta_count + theta;
          const std::size_t output =
              ((mode * point_pipeline_field_count + field) * radial_count +
               radial) *
                  theta_count +
              theta;
          values[output] = point_pipeline_component(points[point], field);
        }
      }
    }
  }
  return values;
}

inline std::vector<PointPipelineState> unpack_point_pipeline_snapshot(
    const std::vector<Complex>& values, const SnapshotShape shape) {
  if (shape.fields != point_pipeline_field_count ||
      values.size() != checked_snapshot_value_count(shape)) {
    throw std::invalid_argument("coupled snapshot values do not match shape");
  }
  const std::size_t radial_count = static_cast<std::size_t>(shape.radial);
  const std::size_t theta_count = static_cast<std::size_t>(shape.theta);
  std::vector<PointPipelineState> points(
      static_cast<std::size_t>(shape.modes) * radial_count * theta_count);
  for (std::size_t mode = 0; mode < shape.modes; ++mode) {
    for (std::size_t field = 0; field < shape.fields; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          const std::size_t point =
              (mode * radial_count + radial) * theta_count + theta;
          const std::size_t input =
              ((mode * point_pipeline_field_count + field) * radial_count +
               radial) *
                  theta_count +
              theta;
          set_point_pipeline_component(points[point], field, values[input]);
        }
      }
    }
  }
  return points;
}

}  // namespace teuk
