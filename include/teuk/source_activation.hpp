#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "teuk/angular.hpp"
#include "teuk/grid.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

struct SourceActivationState {
  bool active = false;
  double activation_time = -1.0;
  int consecutive_passes = 0;
  double last_eligibility_time = -1.0;
};

struct SourceConstraintFamilyNorms {
  double absolute_maximum = 0.0;
  double normalized_maximum = 0.0;
  double weighted_rms = 0.0;
  double normalized_weighted = 0.0;
};

struct SourceConstraintNorms {
  SourceConstraintFamilyNorms psi3_bianchi;
  SourceConstraintFamilyNorms psi2_bianchi;
  SourceConstraintFamilyNorms hll_reality;
  double controlling_normalized = 0.0;
  bool all_finite = true;
};

// Allocation-owning reduction workspace for the three independent
// reconstruction constraints. Only compact scalar reductions cross to the
// host; no stage/state field is copied out of the execution space.
class SourceConstraintEvaluator {
 public:
  SourceConstraintEvaluator(
      const std::size_t mode_count, const UniformRadialGrid& radial_grid,
      const angular::GaussLegendreGrid& angular_grid,
      const std::string& label = "source_constraint_evaluator",
      const RadialDiscretization radial_discretization =
          RadialDiscretization::D42)
      : mode_count_(mode_count),
        radial_count_(radial_grid.size()),
        theta_count_(angular_grid.size()),
        radial_spacing_(radial_grid.spacing()),
        radial_discretization_(radial_discretization),
        weight_sum_(static_cast<double>(mode_count) *
                    (radial_grid.upper_radius() -
                     radial_grid.lower_radius()) *
                    4.0 * angular::pi),
        theta_weights_(label + "_theta_weights", angular_grid.size()),
        weighted_constraint_squared_(label + "_weighted_constraint_squared",
                                     3),
        weighted_left_squared_(label + "_weighted_left_squared", 3),
        weighted_right_squared_(label + "_weighted_right_squared", 3),
        maximum_constraint_squared_(label + "_maximum_constraint_squared", 3),
        maximum_natural_squared_(label + "_maximum_natural_squared", 3),
        maximum_normalized_squared_(label + "_maximum_normalized_squared", 3),
        finite_(label + "_finite", 1),
        host_weighted_constraint_squared_(
            label + "_host_weighted_constraint_squared", 3),
        host_weighted_left_squared_(label + "_host_weighted_left_squared", 3),
        host_weighted_right_squared_(label + "_host_weighted_right_squared", 3),
        host_maximum_constraint_squared_(
            label + "_host_maximum_constraint_squared", 3),
        host_maximum_normalized_squared_(
            label + "_host_maximum_normalized_squared", 3),
        host_finite_(label + "_host_finite", 1) {
    if (mode_count_ == 0 ||
        radial_count_ < radial_minimum_points(radial_discretization_) ||
        theta_count_ == 0 || angular_grid.weights.size() != theta_count_) {
      throw std::invalid_argument(
          "source constraint evaluator requires valid nonempty grids");
    }
    auto host_weights = Kokkos::create_mirror_view(theta_weights_);
    for (std::size_t theta = 0; theta < theta_count_; ++theta) {
      host_weights(theta) = angular_grid.weights[theta];
    }
    Kokkos::deep_copy(theta_weights_, host_weights);
  }

  template <class ExecutionSpace, class ResidualView, class TermView>
  SourceConstraintNorms sample(const ExecutionSpace& execution,
                               const ResidualView& residuals,
                               const TermView& terms) {
    static_assert(ResidualView::rank == 4 && TermView::rank == 4,
                  "source constraint fields must be rank four");
    if (residuals.extent(0) != mode_count_ || residuals.extent(1) != 3 ||
        residuals.extent(2) != radial_count_ ||
        residuals.extent(3) != theta_count_ ||
        terms.extent(0) != mode_count_ || terms.extent(1) != 6 ||
        terms.extent(2) != radial_count_ || terms.extent(3) != theta_count_) {
      throw std::invalid_argument("source constraint reduction extent mismatch");
    }
    Kokkos::deep_copy(execution, weighted_constraint_squared_, 0.0);
    Kokkos::deep_copy(execution, weighted_left_squared_, 0.0);
    Kokkos::deep_copy(execution, weighted_right_squared_, 0.0);
    Kokkos::deep_copy(execution, maximum_constraint_squared_, 0.0);
    Kokkos::deep_copy(execution, maximum_natural_squared_, 0.0);
    Kokkos::deep_copy(execution, maximum_normalized_squared_, 0.0);
    Kokkos::deep_copy(execution, finite_, 1);

    const auto theta_weights = theta_weights_;
    const auto weighted_c = weighted_constraint_squared_;
    const auto weighted_l = weighted_left_squared_;
    const auto weighted_r = weighted_right_squared_;
    const auto maximum_c = maximum_constraint_squared_;
    const auto maximum_natural = maximum_natural_squared_;
    const auto finite = finite_;
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const double spacing = radial_spacing_;
    const RadialDiscretization radial_discretization =
        radial_discretization_;
    const std::size_t total = mode_count_ * radial_count * theta_count;
    Kokkos::parallel_for(
        "reduce_source_constraint_natural_norms",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t theta = flat % theta_count;
          const std::size_t radial = (flat / theta_count) % radial_count;
          const std::size_t mode = flat / (theta_count * radial_count);
          const double weight =
              spacing *
              radial_norm_weight(radial_discretization, radial_count, radial) *
              2.0 * angular::pi * theta_weights(theta);
          for (std::size_t family = 0; family < 3; ++family) {
            const double c = Kokkos::abs(residuals(mode, family, radial, theta));
            const double l = Kokkos::abs(
                terms(mode, 2 * family, radial, theta));
            const double r = Kokkos::abs(
                terms(mode, 2 * family + 1, radial, theta));
            if (!Kokkos::isfinite(c) || !Kokkos::isfinite(l) ||
                !Kokkos::isfinite(r)) {
              Kokkos::atomic_exchange(&finite(0), 0);
              continue;
            }
            Kokkos::atomic_add(&weighted_c(family), weight * c * c);
            Kokkos::atomic_add(&weighted_l(family), weight * l * l);
            Kokkos::atomic_add(&weighted_r(family), weight * r * r);
            Kokkos::atomic_max(&maximum_c(family), c * c);
            Kokkos::atomic_max(&maximum_natural(family),
                               Kokkos::fmax(l * l, r * r));
          }
        });

    const auto maximum_normalized = maximum_normalized_squared_;
    constexpr double relative_floor =
        64.0 * std::numeric_limits<double>::epsilon();
    Kokkos::parallel_for(
        "reduce_source_constraint_normalized_maxima",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t theta = flat % theta_count;
          const std::size_t radial = (flat / theta_count) % radial_count;
          const std::size_t mode = flat / (theta_count * radial_count);
          for (std::size_t family = 0; family < 3; ++family) {
            const double c = Kokkos::abs(residuals(mode, family, radial, theta));
            const double l = Kokkos::abs(
                terms(mode, 2 * family, radial, theta));
            const double r = Kokkos::abs(
                terms(mode, 2 * family + 1, radial, theta));
            const double scale = Kokkos::sqrt(maximum_natural(family));
            const double denominator = l + r + relative_floor * scale;
            const double normalized = denominator > 0.0 ? c / denominator : 0.0;
            if (Kokkos::isfinite(normalized)) {
              Kokkos::atomic_max(&maximum_normalized(family),
                                 normalized * normalized);
            } else {
              Kokkos::atomic_exchange(&finite(0), 0);
            }
          }
        });

    Kokkos::deep_copy(execution, host_weighted_constraint_squared_,
                      weighted_constraint_squared_);
    Kokkos::deep_copy(execution, host_weighted_left_squared_,
                      weighted_left_squared_);
    Kokkos::deep_copy(execution, host_weighted_right_squared_,
                      weighted_right_squared_);
    Kokkos::deep_copy(execution, host_maximum_constraint_squared_,
                      maximum_constraint_squared_);
    Kokkos::deep_copy(execution, host_maximum_normalized_squared_,
                      maximum_normalized_squared_);
    Kokkos::deep_copy(execution, host_finite_, finite_);
    execution.fence("copy normalized source constraint reductions");

    SourceConstraintNorms report;
    report.all_finite = host_finite_(0) != 0;
    std::array<SourceConstraintFamilyNorms*, 3> families{
        &report.psi3_bianchi, &report.psi2_bianchi, &report.hll_reality};
    for (std::size_t family = 0; family < 3; ++family) {
      auto& result = *families[family];
      result.absolute_maximum =
          std::sqrt(host_maximum_constraint_squared_(family));
      result.normalized_maximum =
          std::sqrt(host_maximum_normalized_squared_(family));
      result.weighted_rms =
          std::sqrt(host_weighted_constraint_squared_(family) / weight_sum_);
      const double left =
          std::sqrt(host_weighted_left_squared_(family));
      const double right =
          std::sqrt(host_weighted_right_squared_(family));
      const double natural = std::max(left, right);
      const double denominator =
          left + right + relative_floor * natural;
      result.normalized_weighted =
          denominator > 0.0
              ? std::sqrt(host_weighted_constraint_squared_(family)) /
                    denominator
              : 0.0;
      report.controlling_normalized =
          std::max(report.controlling_normalized,
                   std::max(result.normalized_maximum,
                            result.normalized_weighted));
    }
    if (!report.all_finite || !std::isfinite(report.controlling_normalized)) {
      report.all_finite = false;
      report.controlling_normalized =
          std::numeric_limits<double>::infinity();
    }
    return report;
  }

 private:
  std::size_t mode_count_;
  std::size_t radial_count_;
  std::size_t theta_count_;
  double radial_spacing_;
  RadialDiscretization radial_discretization_;
  double weight_sum_;
  Kokkos::View<double*, MemorySpace> theta_weights_;
  Kokkos::View<double*, MemorySpace> weighted_constraint_squared_;
  Kokkos::View<double*, MemorySpace> weighted_left_squared_;
  Kokkos::View<double*, MemorySpace> weighted_right_squared_;
  Kokkos::View<double*, MemorySpace> maximum_constraint_squared_;
  Kokkos::View<double*, MemorySpace> maximum_natural_squared_;
  Kokkos::View<double*, MemorySpace> maximum_normalized_squared_;
  Kokkos::View<int*, MemorySpace> finite_;
  Kokkos::View<double*, Kokkos::HostSpace>
      host_weighted_constraint_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_weighted_left_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_weighted_right_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_maximum_constraint_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_maximum_normalized_squared_;
  Kokkos::View<int*, Kokkos::HostSpace> host_finite_;
};

}  // namespace teuk
