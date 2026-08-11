#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/reconstruction_diagnostics.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace teuk {

struct IndependentReconstructionConstraintNorms {
  ResidualNorm psi3_bianchi;
  ResidualNorm psi2_bianchi;
  ResidualNorm hll_reality;
  ResidualNorm combined;
};

// Fixed-size host report for the three genuinely independent reconstruction
// checks. SpatialPipeline::evaluate_rhs must have populated the stage-local
// residual View immediately before sampling.
class PipelineIndependentReconstructionDiagnostics {
 public:
  PipelineIndependentReconstructionDiagnostics(
      const std::size_t mode_count, const std::size_t radial_count,
      const std::size_t theta_count,
      const std::string& label = "pipeline_independent_reconstruction")
      : mode_count_(mode_count),
        radial_count_(radial_count),
        theta_count_(theta_count),
        sum_squared_(label + "_sum_squared", 3),
        max_squared_(label + "_max_squared", 3),
        finite_(label + "_finite", 1),
        host_sum_squared_(label + "_host_sum_squared", 3),
        host_max_squared_(label + "_host_max_squared", 3),
        host_finite_(label + "_host_finite", 1) {
    if (mode_count == 0 || radial_count == 0 || theta_count == 0) {
      throw std::invalid_argument(
          "independent reconstruction diagnostics require nonempty extents");
    }
  }

  template <class ExecutionSpace>
  IndependentReconstructionConstraintNorms sample(
      const ExecutionSpace& execution, const SpatialPipeline& pipeline) {
    const auto residuals = pipeline.independent_reconstruction_constraints();
    if (residuals.extent(0) != mode_count_ || residuals.extent(1) != 3 ||
        residuals.extent(2) != radial_count_ ||
        residuals.extent(3) != theta_count_) {
      throw std::invalid_argument(
          "independent reconstruction diagnostic extents mismatch");
    }
    Kokkos::deep_copy(execution, sum_squared_, 0.0);
    Kokkos::deep_copy(execution, max_squared_, 0.0);
    Kokkos::deep_copy(execution, finite_, 1);
    const auto sums = sum_squared_;
    const auto maxima = max_squared_;
    const auto finite = finite_;
    const std::size_t mode_count = mode_count_;
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count * radial_count * theta_count;
    Kokkos::parallel_for(
        "reduce_independent_reconstruction_constraints",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t theta = flat % theta_count;
          const std::size_t radial = (flat / theta_count) % radial_count;
          const std::size_t mode = flat / (theta_count * radial_count);
          for (std::size_t constraint = 0; constraint < 3; ++constraint) {
            const Complex value = residuals(mode, constraint, radial, theta);
            const double magnitude = Kokkos::abs(value);
            if (Kokkos::isfinite(magnitude)) {
              Kokkos::atomic_add(&sums(constraint), magnitude * magnitude);
              Kokkos::atomic_max(&maxima(constraint), magnitude * magnitude);
            } else {
              Kokkos::atomic_exchange(&finite(0), 0);
            }
          }
        });
    Kokkos::deep_copy(execution, host_sum_squared_, sum_squared_);
    Kokkos::deep_copy(execution, host_max_squared_, max_squared_);
    Kokkos::deep_copy(execution, host_finite_, finite_);
    execution.fence("copy independent reconstruction constraints");
    if (host_finite_(0) == 0) {
      throw std::runtime_error("nonfinite independent reconstruction constraint");
    }

    std::array<ResidualNorm, 3> norms{};
    double combined_sum = 0.0;
    double combined_maximum = 0.0;
    for (std::size_t constraint = 0; constraint < 3; ++constraint) {
      norms[constraint] = {
          std::sqrt(host_sum_squared_(constraint) /
                    static_cast<double>(total)),
          std::sqrt(host_max_squared_(constraint))};
      combined_sum += host_sum_squared_(constraint);
      combined_maximum =
          std::max(combined_maximum, host_max_squared_(constraint));
    }
    return {norms[0], norms[1], norms[2],
            {std::sqrt(combined_sum / static_cast<double>(3 * total)),
             std::sqrt(combined_maximum)}};
  }

 private:
  std::size_t mode_count_;
  std::size_t radial_count_;
  std::size_t theta_count_;
  Kokkos::View<double*, MemorySpace> sum_squared_;
  Kokkos::View<double*, MemorySpace> max_squared_;
  Kokkos::View<int*, MemorySpace> finite_;
  Kokkos::View<double*, Kokkos::HostSpace> host_sum_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_max_squared_;
  Kokkos::View<int*, Kokkos::HostSpace> host_finite_;
};

}  // namespace teuk
