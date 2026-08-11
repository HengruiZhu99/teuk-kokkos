#pragma once

#include <Kokkos_Complex.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/types.hpp"

namespace teuk {

// Small repository-local dense complex solve for construction-time Galerkin
// systems. The matrix is row-major and overwritten by partial-pivoting
// Gaussian elimination. This is deliberately not a production large-system
// linear algebra facility.
inline std::vector<Complex> solve_dense_complex_system(
    std::vector<Complex> matrix, std::vector<Complex> rhs) {
  const std::size_t n = rhs.size();
  if (n == 0 || matrix.size() != n * n) {
    throw std::invalid_argument("dense complex system has inconsistent shape");
  }
  double matrix_scale = 0.0;
  for (const Complex value : matrix) {
    matrix_scale = std::max(matrix_scale,
                            static_cast<double>(Kokkos::abs(value)));
  }
  const double pivot_floor =
      128.0 * std::numeric_limits<double>::epsilon() * matrix_scale;
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    double pivot_magnitude = Kokkos::abs(matrix[column * n + column]);
    for (std::size_t row = column + 1; row < n; ++row) {
      const double candidate = Kokkos::abs(matrix[row * n + column]);
      if (candidate > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = candidate;
      }
    }
    if (!std::isfinite(pivot_magnitude) || pivot_magnitude <= pivot_floor) {
      throw std::runtime_error("singular Galerkin initialization matrix");
    }
    if (pivot != column) {
      for (std::size_t j = column; j < n; ++j) {
        std::swap(matrix[column * n + j], matrix[pivot * n + j]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    const Complex diagonal = matrix[column * n + column];
    for (std::size_t row = column + 1; row < n; ++row) {
      const Complex factor = matrix[row * n + column] / diagonal;
      matrix[row * n + column] = Complex(0.0, 0.0);
      for (std::size_t j = column + 1; j < n; ++j) {
        matrix[row * n + j] -= factor * matrix[column * n + j];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  std::vector<Complex> solution(n, Complex(0.0, 0.0));
  for (std::size_t reverse = 0; reverse < n; ++reverse) {
    const std::size_t row = n - 1 - reverse;
    Complex value = rhs[row];
    for (std::size_t column = row + 1; column < n; ++column) {
      value -= matrix[row * n + column] * solution[column];
    }
    solution[row] = value / matrix[row * n + row];
  }
  return solution;
}

inline double dense_complex_relative_residual(
    const std::vector<Complex>& matrix, const std::vector<Complex>& solution,
    const std::vector<Complex>& rhs) {
  const std::size_t n = rhs.size();
  if (n == 0 || solution.size() != n || matrix.size() != n * n) {
    throw std::invalid_argument("dense residual has inconsistent shape");
  }
  double residual_squared = 0.0;
  double rhs_squared = 0.0;
  for (std::size_t row = 0; row < n; ++row) {
    Complex residual = -rhs[row];
    for (std::size_t column = 0; column < n; ++column) {
      residual += matrix[row * n + column] * solution[column];
    }
    residual_squared += Kokkos::abs(residual) * Kokkos::abs(residual);
    rhs_squared += Kokkos::abs(rhs[row]) * Kokkos::abs(rhs[row]);
  }
  return std::sqrt(residual_squared) /
         std::max(std::sqrt(rhs_squared), std::numeric_limits<double>::min());
}

}  // namespace teuk
