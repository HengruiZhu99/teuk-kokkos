#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

namespace teuk {

// Uniform compactified-radius grid.  Storage is ordered from future null
// infinity (R=0 by default) toward the outer horizon.
class UniformRadialGrid {
 public:
  UniformRadialGrid(const std::size_t point_count, const double lower_radius,
                    const double upper_radius)
      : point_count_(point_count),
        lower_radius_(lower_radius),
        upper_radius_(upper_radius),
        spacing_((upper_radius - lower_radius) /
                 static_cast<double>(point_count > 1 ? point_count - 1 : 1)) {
    if (point_count < 5) {
      throw std::invalid_argument(
          "fourth-order radial differences require at least five points");
    }
    if (!(upper_radius > lower_radius)) {
      throw std::invalid_argument(
          "radial grid upper bound must exceed its lower bound");
    }
  }

  explicit UniformRadialGrid(const std::size_t point_count,
                             const double outer_horizon_radius)
      : UniformRadialGrid(point_count, 0.0, outer_horizon_radius) {}

  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t size() const {
    return point_count_;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION double lower_radius() const {
    return lower_radius_;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION double upper_radius() const {
    return upper_radius_;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION double spacing() const {
    return spacing_;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION double coordinate(
      const std::size_t radial_index) const {
    return lower_radius_ + spacing_ * static_cast<double>(radial_index);
  }

 private:
  std::size_t point_count_;
  double lower_radius_;
  double upper_radius_;
  double spacing_;
};

}  // namespace teuk
