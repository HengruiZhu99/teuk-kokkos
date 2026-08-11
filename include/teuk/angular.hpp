#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "teuk/types.hpp"

namespace teuk::angular {

inline constexpr Real pi = std::numbers::pi_v<Real>;

struct GaussLegendreGrid {
  // Nodes x = cos(theta) are strictly increasing from -1 to 1.
  std::vector<Real> x;
  std::vector<Real> weights;

  [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
  [[nodiscard]] Real theta(const std::size_t i) const {
    return std::acos(x.at(i));
  }
};

namespace detail {

inline std::pair<long double, long double> legendre_and_previous(
    const int degree, const long double x) {
  long double p_previous = 1.0L;
  if (degree == 0) return {p_previous, 0.0L};
  long double p = x;
  for (int ell = 2; ell <= degree; ++ell) {
    const long double p_next =
        ((2.0L * ell - 1.0L) * x * p - (ell - 1.0L) * p_previous) /
        ell;
    p_previous = p;
    p = p_next;
  }
  return {p, p_previous};
}

inline long double integer_power(long double base, int exponent) {
  if (exponent < 0) {
    throw std::logic_error("negative exponent in Wigner d evaluation");
  }
  long double result = 1.0L;
  while (exponent != 0) {
    if ((exponent & 1) != 0) result *= base;
    base *= base;
    exponent >>= 1;
  }
  return result;
}

inline long double log_factorial(const int n) {
  if (n < 0) throw std::logic_error("factorial of a negative integer");
  return std::lgammal(static_cast<long double>(n) + 1.0L);
}

}  // namespace detail

// Golub-Welsch is unnecessary for the modest angular grids used here. Newton
// iteration on Legendre roots is dependency-free, deterministic, and obtains
// the standard 2/(1-x^2)P'_n(x)^2 weights.
inline GaussLegendreGrid gauss_legendre(const int node_count) {
  if (node_count <= 0) {
    throw std::invalid_argument("Gauss-Legendre node count must be positive");
  }

  GaussLegendreGrid grid;
  grid.x.resize(static_cast<std::size_t>(node_count));
  grid.weights.resize(static_cast<std::size_t>(node_count));
  const int roots_to_find = (node_count + 1) / 2;
  const long double tolerance =
      8.0L * std::numeric_limits<long double>::epsilon();

  for (int i = 0; i < roots_to_find; ++i) {
    long double root = std::cos(std::numbers::pi_v<long double> *
                                (i + 0.75L) / (node_count + 0.5L));
    long double derivative = 0.0L;
    for (int iteration = 0; iteration < 100; ++iteration) {
      const auto [pn, pn_minus_one] =
          detail::legendre_and_previous(node_count, root);
      derivative = node_count * (root * pn - pn_minus_one) /
                   (root * root - 1.0L);
      const long double update = pn / derivative;
      root -= update;
      if (std::abs(update) <= tolerance * (1.0L + std::abs(root))) break;
      if (iteration == 99) {
        throw std::runtime_error("Gauss-Legendre root iteration failed");
      }
    }

    // Refresh the derivative at the converged root before computing the weight.
    const auto [pn, pn_minus_one] =
        detail::legendre_and_previous(node_count, root);
    (void)pn;
    derivative = node_count * (root * pn - pn_minus_one) /
                 (root * root - 1.0L);
    const long double weight =
        2.0L / ((1.0L - root * root) * derivative * derivative);

    const auto low = static_cast<std::size_t>(i);
    const auto high = static_cast<std::size_t>(node_count - 1 - i);
    grid.x[low] = static_cast<Real>(-root);
    grid.x[high] = static_cast<Real>(root);
    grid.weights[low] = static_cast<Real>(weight);
    grid.weights[high] = static_cast<Real>(weight);
  }
  return grid;
}

inline void validate_mode(const int ell, const int spin, const int m) {
  if (ell < 0 || std::abs(spin) > ell || std::abs(m) > ell) {
    throw std::invalid_argument("spin-weighted mode requires ell >= |s|,|m|");
  }
}

[[nodiscard]] inline int minimum_ell(const int spin, const int m) noexcept {
  return std::max(std::abs(spin), std::abs(m));
}

// Unit-sphere eth/raising and eth-prime/lowering factors fixed by
// docs/CONVENTIONS.md and the corrected implementer reference.
[[nodiscard]] inline Real raising_factor(const int ell, const int spin) {
  validate_mode(ell, spin, 0);
  return std::sqrt(static_cast<Real>((ell - spin) * (ell + spin + 1)));
}

[[nodiscard]] inline Real lowering_factor(const int ell, const int spin) {
  validate_mode(ell, spin, 0);
  return -std::sqrt(static_cast<Real>((ell + spin) * (ell - spin + 1)));
}

[[nodiscard]] inline Real lower_after_raise_eigenvalue(const int ell,
                                                       const int spin) {
  validate_mode(ell, spin, 0);
  return -static_cast<Real>((ell - spin) * (ell + spin + 1));
}

[[nodiscard]] inline Real raise_after_lower_eigenvalue(const int ell,
                                                       const int spin) {
  validate_mode(ell, spin, 0);
  return -static_cast<Real>((ell + spin) * (ell - spin + 1));
}

// The spin-covariant unit-sphere Laplacian is the symmetric composition
// (L_{s+1}R_s + R_{s-1}L_s)/2. Its eigenvalue is s^2-l(l+1).
[[nodiscard]] inline Real spin_weighted_laplacian_eigenvalue(const int ell,
                                                             const int spin) {
  validate_mode(ell, spin, 0);
  return static_cast<Real>(spin * spin - ell * (ell + 1));
}

// Wigner small-d matrix in the Condon-Shortley convention. Long-double
// accumulation makes this compact reference routine useful at modest
// bandlimits; optimized production transforms can be checked against it.
[[nodiscard]] inline Real wigner_small_d(const int ell, const int m_prime,
                                         const int m, const Real theta) {
  validate_mode(ell, 0, m_prime);
  validate_mode(ell, 0, m);
  if (theta < 0.0 || theta > pi) {
    throw std::invalid_argument("theta must lie in [0, pi]");
  }

  const int k_min = std::max(0, m - m_prime);
  const int k_max = std::min(ell + m, ell - m_prime);
  const long double log_normalization =
      0.5L * (detail::log_factorial(ell + m) +
              detail::log_factorial(ell - m) +
              detail::log_factorial(ell + m_prime) +
              detail::log_factorial(ell - m_prime));
  const long double cosine =
      std::cos(0.5L * static_cast<long double>(theta));
  const long double sine =
      std::sin(0.5L * static_cast<long double>(theta));
  long double value = 0.0L;

  for (int k = k_min; k <= k_max; ++k) {
    const long double log_coefficient =
        log_normalization - detail::log_factorial(ell + m - k) -
        detail::log_factorial(k) -
        detail::log_factorial(m_prime - m + k) -
        detail::log_factorial(ell - m_prime - k);
    const int cosine_power = 2 * ell + m - m_prime - 2 * k;
    const int sine_power = m_prime - m + 2 * k;
    long double term = std::exp(log_coefficient) *
                       detail::integer_power(cosine, cosine_power) *
                       detail::integer_power(sine, sine_power);
    if (((m_prime - m + k) & 1) != 0) term = -term;
    value += term;
  }
  return static_cast<Real>(value);
}

// Theta-dependent amplitude in
//   {}_sY_lm(theta,phi) = spin_weighted_harmonic_theta(...) exp(i m phi).
// This convention gives Y_11 = -sqrt(3/(8 pi)) sin(theta) exp(i phi).
[[nodiscard]] inline Real spin_weighted_harmonic_theta(
    const int ell, const int m, const int spin, const Real theta) {
  validate_mode(ell, spin, m);
  const Real phase = (std::abs(spin) & 1) != 0 ? -1.0 : 1.0;
  return phase * std::sqrt((2.0 * ell + 1.0) / (4.0 * pi)) *
         wigner_small_d(ell, m, -spin, theta);
}

// Integer Wigner 3-j symbol evaluated with the Racah factorial sum. Invalid
// magnetic or triangle couplings are exact selection-rule zeros. This compact
// long-double oracle is intended for the modest angular bandlimits used by the
// readable reference path, not as a large-j special-functions replacement.
[[nodiscard]] inline Real wigner_3j(const int j1, const int j2, const int j3,
                                    const int m1, const int m2,
                                    const int m3) {
  if (j1 < 0 || j2 < 0 || j3 < 0) {
    throw std::invalid_argument("Wigner 3-j angular momenta must be nonnegative");
  }
  if (m1 + m2 + m3 != 0 || std::abs(m1) > j1 || std::abs(m2) > j2 ||
      std::abs(m3) > j3 || j3 > j1 + j2 || j3 < std::abs(j1 - j2)) {
    return 0.0;
  }

  const long double log_triangle =
      detail::log_factorial(j1 + j2 - j3) +
      detail::log_factorial(j1 - j2 + j3) +
      detail::log_factorial(-j1 + j2 + j3) -
      detail::log_factorial(j1 + j2 + j3 + 1);
  const long double log_magnetic_factorials =
      detail::log_factorial(j1 + m1) + detail::log_factorial(j1 - m1) +
      detail::log_factorial(j2 + m2) + detail::log_factorial(j2 - m2) +
      detail::log_factorial(j3 + m3) + detail::log_factorial(j3 - m3);
  const long double prefactor =
      (((j1 - j2 - m3) & 1) != 0 ? -1.0L : 1.0L) *
      std::exp(0.5L * (log_triangle + log_magnetic_factorials));

  const int z_min =
      std::max({0, j2 - j3 - m1, j1 - j3 + m2});
  const int z_max =
      std::min({j1 + j2 - j3, j1 - m1, j2 + m2});
  long double sum = 0.0L;
  for (int z = z_min; z <= z_max; ++z) {
    const long double log_denominator =
        detail::log_factorial(z) +
        detail::log_factorial(j1 + j2 - j3 - z) +
        detail::log_factorial(j1 - m1 - z) +
        detail::log_factorial(j2 + m2 - z) +
        detail::log_factorial(j3 - j2 + m1 + z) +
        detail::log_factorial(j3 - j1 - m2 + z);
    const long double term = std::exp(-log_denominator);
    sum += (z & 1) != 0 ? -term : term;
  }
  return static_cast<Real>(prefactor * sum);
}

[[nodiscard]] inline bool gaunt_selection_allowed(
    const int ell1, const int m1, const int spin1, const int ell2,
    const int m2, const int spin2, const int ell3, const int m3,
    const int spin3) noexcept {
  if (ell1 < 0 || ell2 < 0 || ell3 < 0) return false;
  if (std::abs(m1) > ell1 || std::abs(spin1) > ell1 ||
      std::abs(m2) > ell2 || std::abs(spin2) > ell2 ||
      std::abs(m3) > ell3 || std::abs(spin3) > ell3) {
    return false;
  }
  return m3 == m1 + m2 && spin3 == spin1 + spin2 &&
         ell3 >= std::abs(ell1 - ell2) && ell3 <= ell1 + ell2;
}

// Integral of two spin-weighted harmonics against the conjugate of a third,
// using exactly the convention fixed in the corrected implementer reference.
[[nodiscard]] inline Real spin_weighted_gaunt(
    const int ell1, const int m1, const int spin1, const int ell2,
    const int m2, const int spin2, const int ell3, const int m3,
    const int spin3) {
  if (!gaunt_selection_allowed(ell1, m1, spin1, ell2, m2, spin2,
                               ell3, m3, spin3)) {
    return 0.0;
  }
  const Real phase = ((m3 + spin3) & 1) != 0 ? -1.0 : 1.0;
  const Real normalization =
      std::sqrt(static_cast<Real>((2 * ell1 + 1) * (2 * ell2 + 1) *
                                  (2 * ell3 + 1)) /
                (4.0 * pi));
  return phase * normalization *
         wigner_3j(ell1, ell2, ell3, -spin1, -spin2, spin3) *
         wigner_3j(ell1, ell2, ell3, m1, m2, -m3);
}

class DenseRealMatrix {
 public:
  DenseRealMatrix() = default;
  DenseRealMatrix(const std::size_t rows, const std::size_t columns)
      : rows_(rows), columns_(columns), entries_(rows * columns, 0.0) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns() const noexcept { return columns_; }

  Real& operator()(const std::size_t row, const std::size_t column) noexcept {
    return entries_[row * columns_ + column];
  }
  [[nodiscard]] Real operator()(const std::size_t row,
                                const std::size_t column) const noexcept {
    return entries_[row * columns_ + column];
  }
  [[nodiscard]] const std::vector<Real>& entries() const noexcept {
    return entries_;
  }

 private:
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::vector<Real> entries_;
};

// Small, transparent fixed-(s,m) modal/collocation reference. The synthesis
// matrix is node-by-ell and the analysis matrix includes the full-sphere 2*pi
// factor and Gauss-Legendre weights.
class SpinWeightedTransform {
 public:
  SpinWeightedTransform(const int spin, const int m, const int ell_max,
                        int node_count = 0)
      : spin_(spin), m_(m), ell_min_(minimum_ell(spin, m)), ell_max_(ell_max) {
    if (ell_max_ < ell_min_) {
      throw std::invalid_argument("ell_max is below the minimum allowed ell");
    }
    if (node_count == 0) node_count = ell_max_ + 1;
    if (static_cast<std::size_t>(node_count) < mode_count()) {
      throw std::invalid_argument(
          "angular transform needs at least as many nodes as modes");
    }
    grid_ = gauss_legendre(node_count);
    synthesis_ = DenseRealMatrix(grid_.size(), mode_count());
    analysis_ = DenseRealMatrix(mode_count(), grid_.size());
    for (std::size_t node = 0; node < grid_.size(); ++node) {
      const Real theta = grid_.theta(node);
      for (std::size_t mode = 0; mode < mode_count(); ++mode) {
        const int ell = ell_min_ + static_cast<int>(mode);
        const Real basis =
            spin_weighted_harmonic_theta(ell, m_, spin_, theta);
        synthesis_(node, mode) = basis;
        analysis_(mode, node) = 2.0 * pi * grid_.weights[node] * basis;
      }
    }
  }

  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int m() const noexcept { return m_; }
  [[nodiscard]] int ell_min() const noexcept { return ell_min_; }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }
  [[nodiscard]] std::size_t mode_count() const noexcept {
    return static_cast<std::size_t>(ell_max_ - ell_min_ + 1);
  }
  [[nodiscard]] const GaussLegendreGrid& grid() const noexcept { return grid_; }
  [[nodiscard]] const DenseRealMatrix& synthesis_matrix() const noexcept {
    return synthesis_;
  }
  [[nodiscard]] const DenseRealMatrix& analysis_matrix() const noexcept {
    return analysis_;
  }

  [[nodiscard]] std::vector<Complex> synthesize(
      const std::vector<Complex>& modal) const {
    if (modal.size() != mode_count()) {
      throw std::invalid_argument("wrong modal vector size");
    }
    std::vector<Complex> nodal(grid_.size(), Complex(0.0, 0.0));
    for (std::size_t node = 0; node < grid_.size(); ++node) {
      for (std::size_t mode = 0; mode < mode_count(); ++mode) {
        nodal[node] += synthesis_(node, mode) * modal[mode];
      }
    }
    return nodal;
  }

  [[nodiscard]] std::vector<Complex> analyze(
      const std::vector<Complex>& nodal) const {
    if (nodal.size() != grid_.size()) {
      throw std::invalid_argument("wrong nodal vector size");
    }
    std::vector<Complex> modal(mode_count(), Complex(0.0, 0.0));
    for (std::size_t mode = 0; mode < mode_count(); ++mode) {
      for (std::size_t node = 0; node < grid_.size(); ++node) {
        modal[mode] += analysis_(mode, node) * nodal[node];
      }
    }
    return modal;
  }

  [[nodiscard]] std::vector<Complex> raise(
      const std::vector<Complex>& modal) const {
    return apply_diagonal(modal, true);
  }

  [[nodiscard]] std::vector<Complex> lower(
      const std::vector<Complex>& modal) const {
    return apply_diagonal(modal, false);
  }

  [[nodiscard]] std::vector<Complex> laplacian(
      const std::vector<Complex>& modal) const {
    if (modal.size() != mode_count()) {
      throw std::invalid_argument("wrong modal vector size");
    }
    std::vector<Complex> result(mode_count());
    for (std::size_t mode = 0; mode < mode_count(); ++mode) {
      const int ell = ell_min_ + static_cast<int>(mode);
      result[mode] = spin_weighted_laplacian_eigenvalue(ell, spin_) *
                     modal[mode];
    }
    return result;
  }

 private:
  [[nodiscard]] std::vector<Complex> apply_diagonal(
      const std::vector<Complex>& modal, const bool raising) const {
    if (modal.size() != mode_count()) {
      throw std::invalid_argument("wrong modal vector size");
    }
    std::vector<Complex> result(mode_count());
    for (std::size_t mode = 0; mode < mode_count(); ++mode) {
      const int ell = ell_min_ + static_cast<int>(mode);
      const Real factor = raising ? raising_factor(ell, spin_)
                                  : lowering_factor(ell, spin_);
      result[mode] = factor * modal[mode];
    }
    return result;
  }

  int spin_;
  int m_;
  int ell_min_;
  int ell_max_;
  GaussLegendreGrid grid_;
  DenseRealMatrix synthesis_;
  DenseRealMatrix analysis_;
};

inline void validate_product_metadata(const SpinWeightedTransform& left,
                                      const SpinWeightedTransform& right,
                                      const SpinWeightedTransform& target) {
  if (target.spin() != left.spin() + right.spin()) {
    throw std::invalid_argument("product target spin is not s1+s2");
  }
  if (target.m() != left.m() + right.m()) {
    throw std::invalid_argument("product target mode is not m1+m2");
  }
}

// Exact truncated modal convolution. Each returned coefficient is the full
// spherical projection sum over the two supplied bands.
[[nodiscard]] inline std::vector<Complex> exact_modal_product(
    const SpinWeightedTransform& left,
    const std::vector<Complex>& left_modal,
    const SpinWeightedTransform& right,
    const std::vector<Complex>& right_modal,
    const SpinWeightedTransform& target) {
  validate_product_metadata(left, right, target);
  if (left_modal.size() != left.mode_count() ||
      right_modal.size() != right.mode_count()) {
    throw std::invalid_argument("wrong source modal vector size");
  }

  std::vector<Complex> result(target.mode_count(), Complex(0.0, 0.0));
  for (std::size_t target_mode = 0; target_mode < target.mode_count();
       ++target_mode) {
    const int ell3 = target.ell_min() + static_cast<int>(target_mode);
    for (std::size_t left_mode = 0; left_mode < left.mode_count();
         ++left_mode) {
      const int ell1 = left.ell_min() + static_cast<int>(left_mode);
      for (std::size_t right_mode = 0; right_mode < right.mode_count();
           ++right_mode) {
        const int ell2 = right.ell_min() + static_cast<int>(right_mode);
        const Real coefficient = spin_weighted_gaunt(
            ell1, left.m(), left.spin(), ell2, right.m(), right.spin(), ell3,
            target.m(), target.spin());
        result[target_mode] +=
            coefficient * left_modal[left_mode] * right_modal[right_mode];
      }
    }
  }
  return result;
}

// A Gauss-Legendre rule with 2N-1 >= ell1_max+ell2_max+ell3_max integrates
// every retained triple-harmonic projection in the supplied bands. This is the
// minimum polynomial-exact padding rule; callers may overcollocate further.
[[nodiscard]] inline int padded_product_node_count(
    const SpinWeightedTransform& left, const SpinWeightedTransform& right,
    const SpinWeightedTransform& target) noexcept {
  const int polynomial_count =
      (left.ell_max() + right.ell_max() + target.ell_max() + 2) / 2;
  const int storage_count = static_cast<int>(
      std::max({left.mode_count(), right.mode_count(), target.mode_count()}));
  return std::max(polynomial_count, storage_count);
}

// Independent nonlinear reference path: resynthesize both factors on a common
// grid, multiply pointwise, and project into the target band. Passing zero uses
// the exactness-based padding above; an explicit smaller count is useful only
// for aliasing tests and diagnostics.
[[nodiscard]] inline std::vector<Complex> collocation_product(
    const SpinWeightedTransform& left,
    const std::vector<Complex>& left_modal,
    const SpinWeightedTransform& right,
    const std::vector<Complex>& right_modal,
    const SpinWeightedTransform& target, int node_count = 0) {
  validate_product_metadata(left, right, target);
  if (left_modal.size() != left.mode_count() ||
      right_modal.size() != right.mode_count()) {
    throw std::invalid_argument("wrong source modal vector size");
  }
  if (node_count == 0) {
    node_count = padded_product_node_count(left, right, target);
  }

  const SpinWeightedTransform padded_left(left.spin(), left.m(),
                                          left.ell_max(), node_count);
  const SpinWeightedTransform padded_right(right.spin(), right.m(),
                                           right.ell_max(), node_count);
  const SpinWeightedTransform padded_target(target.spin(), target.m(),
                                            target.ell_max(), node_count);
  const auto left_nodal = padded_left.synthesize(left_modal);
  const auto right_nodal = padded_right.synthesize(right_modal);
  std::vector<Complex> product_nodal(left_nodal.size());
  for (std::size_t node = 0; node < product_nodal.size(); ++node) {
    product_nodal[node] = left_nodal[node] * right_nodal[node];
  }
  return padded_target.analyze(product_nodal);
}

}  // namespace teuk::angular
