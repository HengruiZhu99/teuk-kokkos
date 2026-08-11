#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/grid.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr std::size_t horizon_derivative_count = 5;
inline constexpr std::size_t horizon_stencil_points = 9;

using HorizonWeightView = Kokkos::View<Real**, Kokkos::LayoutRight, MemorySpace>;
using HorizonDerivativeView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;

namespace horizon_detail {

inline long double factorial(const std::size_t value) {
  long double result = 1.0L;
  for (std::size_t factor = 2; factor <= value; ++factor) {
    result *= static_cast<long double>(factor);
  }
  return result;
}

// Solve the moment equations sum_j w_j (-j)^k = k! delta_{kd}. This creates
// a readable, dependency-free backward endpoint stencil exact for every
// polynomial through degree eight.
inline std::array<Real, horizon_stencil_points> backward_weights(
    const std::size_t derivative, const Real spacing) {
  if (derivative >= horizon_derivative_count || !(spacing > 0.0)) {
    throw std::invalid_argument("invalid horizon derivative request");
  }
  constexpr std::size_t n = horizon_stencil_points;
  std::array<std::array<long double, n + 1>, n> matrix{};
  for (std::size_t power = 0; power < n; ++power) {
    for (std::size_t point = 0; point < n; ++point) {
      const long double coordinate = -static_cast<long double>(point);
      long double monomial = 1.0L;
      for (std::size_t exponent = 0; exponent < power; ++exponent) {
        monomial *= coordinate;
      }
      matrix[power][point] = monomial;
    }
    matrix[power][n] =
        power == derivative ? factorial(derivative) : 0.0L;
  }
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row) {
      if (std::abs(matrix[row][column]) >
          std::abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    if (std::abs(matrix[pivot][column]) < 1.0e-28L) {
      throw std::runtime_error("singular horizon stencil moment matrix");
    }
    if (pivot != column) std::swap(matrix[pivot], matrix[column]);
    const long double diagonal = matrix[column][column];
    for (std::size_t entry = column; entry <= n; ++entry) {
      matrix[column][entry] /= diagonal;
    }
    for (std::size_t row = 0; row < n; ++row) {
      if (row == column) continue;
      const long double multiple = matrix[row][column];
      for (std::size_t entry = column; entry <= n; ++entry) {
        matrix[row][entry] -= multiple * matrix[column][entry];
      }
    }
  }
  std::array<Real, n> weights{};
  const long double scale =
      std::pow(static_cast<long double>(spacing),
               -static_cast<int>(derivative));
  for (std::size_t point = 0; point < n; ++point) {
    weights[point] = static_cast<Real>(matrix[point][n] * scale);
  }
  return weights;
}

}  // namespace horizon_detail

// Reusable device evaluator for the first four transverse radial derivatives
// of one field at R=R_H. The result ordering is (mode, derivative, theta),
// where derivative zero is the horizon field value itself.
class HorizonTransverseDiagnostics {
 public:
  HorizonTransverseDiagnostics(
      const UniformRadialGrid& grid, const std::size_t mode_count,
      const std::size_t theta_count,
      const std::string& label = "horizon_transverse")
      : radial_count_(grid.size()),
        mode_count_(mode_count),
        theta_count_(theta_count),
        weights_(label + "_weights", horizon_derivative_count,
                 horizon_stencil_points),
        derivatives_(label + "_derivatives", mode_count,
                     horizon_derivative_count, theta_count) {
    if (radial_count_ < horizon_stencil_points || mode_count_ == 0 ||
        theta_count_ == 0) {
      throw std::invalid_argument(
          "horizon diagnostics require nonzero modes/theta and nine radial points");
    }
    auto host_weights = Kokkos::create_mirror_view(weights_);
    for (std::size_t derivative = 0;
         derivative < horizon_derivative_count; ++derivative) {
      const auto row =
          horizon_detail::backward_weights(derivative, grid.spacing());
      for (std::size_t point = 0; point < horizon_stencil_points; ++point) {
        host_weights(derivative, point) = row[point];
      }
    }
    Kokkos::deep_copy(weights_, host_weights);
  }

  template <class ExecutionSpace, class StateView>
  void evaluate(const ExecutionSpace& execution, const StateView& state,
                const std::size_t field) {
    static_assert(StateView::rank == 4,
                  "horizon diagnostic input must have rank four");
    if (state.extent(0) != mode_count_ || state.extent(1) <= field ||
        state.extent(2) != radial_count_ ||
        state.extent(3) != theta_count_) {
      throw std::invalid_argument("horizon diagnostic state extents mismatch");
    }
    const auto weights = weights_;
    const auto derivatives = derivatives_;
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const std::size_t total =
        mode_count_ * horizon_derivative_count * theta_count_;
    Kokkos::parallel_for(
        "teuk_horizon_transverse_derivatives",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t mode_derivative = flat / theta_count;
          const std::size_t theta = flat - mode_derivative * theta_count;
          const std::size_t mode =
              mode_derivative / horizon_derivative_count;
          const std::size_t derivative =
              mode_derivative - mode * horizon_derivative_count;
          Complex value(0.0, 0.0);
          for (std::size_t point = 0; point < horizon_stencil_points;
               ++point) {
            value += weights(derivative, point) *
                     state(mode, field, radial_count - 1 - point, theta);
          }
          derivatives(mode, derivative, theta) = value;
        });
  }

  [[nodiscard]] HorizonDerivativeView values() const { return derivatives_; }

 private:
  std::size_t radial_count_;
  std::size_t mode_count_;
  std::size_t theta_count_;
  HorizonWeightView weights_;
  HorizonDerivativeView derivatives_;
};

}  // namespace teuk
