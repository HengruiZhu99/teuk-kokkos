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

// Device reduction of the seven independent reconstruction equation
// residuals. SpatialPipeline::evaluate_rhs must have been called for the same
// state immediately before sampling so its stage-local radial and angular
// diagnostic inputs correspond to the reported RHS.
class PipelineReconstructionDiagnostics {
 public:
  PipelineReconstructionDiagnostics(
      const std::size_t mode_count, const std::size_t radial_count,
      const std::size_t theta_count,
      const std::string& label = "pipeline_reconstruction_residual")
      : mode_count_(mode_count),
        radial_count_(radial_count),
        theta_count_(theta_count),
        sum_squared_(label + "_sum_squared", 7),
        max_squared_(label + "_max_squared", 7),
        finite_(label + "_finite", 1),
        host_sum_squared_(label + "_host_sum_squared", 7),
        host_max_squared_(label + "_host_max_squared", 7),
        host_finite_(label + "_host_finite", 1) {
    if (mode_count == 0 || radial_count == 0 || theta_count == 0) {
      throw std::invalid_argument(
          "reconstruction diagnostics require nonempty extents");
    }
  }

  template <class ExecutionSpace>
  ReconstructionResidualNorms sample(const ExecutionSpace& execution,
                                     const SpatialPipeline& pipeline) {
    const auto state = pipeline.storage().state();
    const auto rhs = pipeline.storage().rhs();
    const auto radial = pipeline.reconstruction_radial_derivatives();
    const auto angular = pipeline.reconstruction_angular_inputs();
    const auto sharp = pipeline.storage().sharp_indices();
    const auto radius = pipeline.storage().radius();
    const auto cos_theta = pipeline.storage().cos_theta();
    const auto sin_theta = pipeline.storage().sin_theta();
    if (state.extent(0) != mode_count_ ||
        state.extent(1) != point_pipeline_field_count ||
        state.extent(2) != radial_count_ ||
        state.extent(3) != theta_count_ || radial.extent(0) != mode_count_ ||
        radial.extent(1) != 7 || radial.extent(2) != radial_count_ ||
        radial.extent(3) != theta_count_ || angular.extent(0) != mode_count_ ||
        angular.extent(1) !=
            static_cast<std::size_t>(ReconstructionAngularInput::Count) ||
        angular.extent(2) != radial_count_ ||
        angular.extent(3) != theta_count_) {
      throw std::invalid_argument(
          "reconstruction diagnostic pipeline extents mismatch");
    }

    Kokkos::deep_copy(execution, sum_squared_, 0.0);
    Kokkos::deep_copy(execution, max_squared_, 0.0);
    Kokkos::deep_copy(execution, finite_, 1);
    const auto sums = sum_squared_;
    const auto maxima = max_squared_;
    const auto finite = finite_;
    const KerrParameters parameters = pipeline.background();
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count_ * radial_count_ * theta_count_;
    Kokkos::parallel_for(
        "teuk_pipeline_reconstruction_residuals",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t mode_radial = flat / theta_count;
          const std::size_t theta = flat - mode_radial * theta_count;
          const std::size_t mode = mode_radial / radial_count;
          const std::size_t radial_index =
              mode_radial - mode * radial_count;
          const std::size_t sharp_mode = sharp(mode);
          const double coordinate = radius(radial_index);
          const auto background = kerr_background_point(
              parameters, coordinate, cos_theta(theta), sin_theta(theta));
          const auto value = [&](const PipelineField field) {
            return state(mode, static_cast<std::size_t>(field), radial_index,
                         theta);
          };
          const auto sharp_value = [&](const PipelineField field) {
            return Kokkos::conj(state(sharp_mode,
                                      static_cast<std::size_t>(field),
                                      radial_index, theta));
          };
          const ReconstructionFields fields{
              value(PipelineField::FirstPsi), value(PipelineField::G),
              value(PipelineField::H), value(PipelineField::Lambda),
              value(PipelineField::Pi), value(PipelineField::B),
              value(PipelineField::C), value(PipelineField::U),
              sharp_value(PipelineField::Pi), sharp_value(PipelineField::B),
              sharp_value(PipelineField::C)};
          const ReconstructionAngularDerivatives angular_values{
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::Eth1F),
                      radial_index, theta),
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::Eth2G),
                      radial_index, theta),
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::Eth2C),
                      radial_index, theta),
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::Eth2Pi),
                      radial_index, theta),
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::EthPrime1BSharp),
                      radial_index, theta),
              angular(mode, static_cast<std::size_t>(
                                ReconstructionAngularInput::EthPrime2CSharp),
                      radial_index, theta)};
          const ReconstructionFieldDerivatives time_derivatives{
              rhs(mode, static_cast<std::size_t>(PipelineField::G),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::Lambda),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::H),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::B),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::Pi),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::C),
                  radial_index, theta),
              rhs(mode, static_cast<std::size_t>(PipelineField::U),
                  radial_index, theta)};
          const ReconstructionFieldDerivatives radial_derivatives{
              radial(mode, 0, radial_index, theta),
              radial(mode, 1, radial_index, theta),
              radial(mode, 2, radial_index, theta),
              radial(mode, 3, radial_index, theta),
              radial(mode, 4, radial_index, theta),
              radial(mode, 5, radial_index, theta),
              radial(mode, 6, radial_index, theta)};
          const ReconstructionResiduals residual =
              reconstruction_residuals_point(
                  coordinate, parameters.mass,
                  parameters.compactification_length, background, fields,
                  angular_values, time_derivatives, radial_derivatives);
          const Complex values[7] = {residual.G,  residual.Lambda, residual.H,
                                     residual.B,  residual.Pi,     residual.C,
                                     residual.U};
          for (std::size_t field = 0; field < 7; ++field) {
            const double magnitude = Kokkos::abs(values[field]);
            if (Kokkos::isfinite(magnitude)) {
              Kokkos::atomic_add(&sums(field), magnitude * magnitude);
              Kokkos::atomic_max(&maxima(field), magnitude * magnitude);
            } else {
              Kokkos::atomic_exchange(&finite(0), 0);
            }
          }
        });
    Kokkos::deep_copy(execution, host_sum_squared_, sum_squared_);
    Kokkos::deep_copy(execution, host_max_squared_, max_squared_);
    Kokkos::deep_copy(execution, host_finite_, finite_);
    execution.fence("copy reconstruction residual report");
    if (host_finite_(0) == 0) {
      throw std::runtime_error("nonfinite reconstruction residual");
    }
    std::array<ResidualNorm, 7> norm{};
    double combined_sum = 0.0;
    double combined_maximum = 0.0;
    for (std::size_t field = 0; field < 7; ++field) {
      norm[field] = {
          std::sqrt(host_sum_squared_(field) / static_cast<double>(total)),
          std::sqrt(host_max_squared_(field))};
      combined_sum += host_sum_squared_(field);
      combined_maximum =
          std::max(combined_maximum, host_max_squared_(field));
    }
    const ResidualNorm combined{
        std::sqrt(combined_sum / static_cast<double>(7 * total)),
        std::sqrt(combined_maximum)};
    return {norm[0], norm[1], norm[2], norm[3],
            norm[4], norm[5], norm[6], combined};
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
