#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include "teuk/angular_device.hpp"
#include "teuk/background.hpp"
#include "teuk/ghp_device.hpp"
#include "teuk/modes.hpp"

namespace teuk {

// Host-owned launch coordinator for one field metadata tuple (spin, boost,
// ell_max) over every signed m in a ModeRegistry. All Kokkos Views are allocated
// during construction. Launch methods only create shallow subviews and enqueue
// kernels in sorted registry order on the caller's execution-space instance.
template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class SignedModeAngularCoordinator {
 public:
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using angular_plan = DeviceAngularPlan<execution_space>;
  using ghp_plan = DeviceGhpAngularPlan<execution_space>;
  using real_view = Kokkos::View<Real*, memory_space>;

 private:
  struct ModeResources {
    ModeResources(const execution_space& execution, const int spin,
                  const int m, const int boost, const int ell_max,
                  const int node_count, const std::size_t radial_count,
                  const KerrParameters& parameters)
        : m(m),
          angular(execution, spin, m, ell_max, node_count),
          ghp(execution, spin, m, boost, ell_max, node_count, parameters),
          ghp_workspace(ghp, radial_count),
          laplacian_modal("coordinator_laplacian_modal", radial_count,
                          angular.mode_count()),
          laplacian_operated("coordinator_laplacian_operated", radial_count,
                             angular.mode_count()) {}

    int m;
    angular_plan angular;
    ghp_plan ghp;
    typename ghp_plan::Workspace ghp_workspace;
    typename angular_plan::modal_view laplacian_modal;
    typename angular_plan::modal_view laplacian_operated;
  };

 public:
  SignedModeAngularCoordinator(
      const execution_space& execution, const ModeRegistry& registry,
      const int spin, const int boost, const int ell_max,
      const int node_count, const std::size_t radial_count,
      const KerrParameters& parameters)
      : SignedModeAngularCoordinator(execution, registry, spin, boost, ell_max,
                                     node_count, radial_count, parameters,
                                     registry.modes()) {}

  SignedModeAngularCoordinator(
      const execution_space& execution, const ModeRegistry& registry,
      const int spin, const int boost, const int ell_max,
      const int node_count, const std::size_t radial_count,
      const KerrParameters& parameters, std::vector<int> active_modes)
      : registry_(registry),
        spin_(spin),
        boost_(boost),
        ell_max_(ell_max),
        node_count_(node_count),
        radial_count_(radial_count) {
    if (radial_count == 0) {
      throw std::invalid_argument(
          "angular coordinator requires at least one radial point");
    }
    std::sort(active_modes.begin(), active_modes.end());
    if (std::adjacent_find(active_modes.begin(), active_modes.end()) !=
        active_modes.end()) {
      throw std::invalid_argument("angular coordinator active modes repeat");
    }
    for (const int m : active_modes) {
      if (!registry_.contains(m) || std::abs(m) > ell_max_) {
        throw std::invalid_argument(
            "angular coordinator active mode is outside its stored band");
      }
    }
    resources_.reserve(registry_.size());
    for (const int m : registry_.modes()) {
      if (std::binary_search(active_modes.begin(), active_modes.end(), m)) {
        resources_.push_back(std::make_unique<ModeResources>(
            execution, spin, m, boost, ell_max, node_count, radial_count,
            parameters));
      } else {
        resources_.push_back(nullptr);
      }
    }
  }

  SignedModeAngularCoordinator(const SignedModeAngularCoordinator&) = delete;
  SignedModeAngularCoordinator& operator=(
      const SignedModeAngularCoordinator&) = delete;
  SignedModeAngularCoordinator(SignedModeAngularCoordinator&&) noexcept =
      default;
  SignedModeAngularCoordinator& operator=(
      SignedModeAngularCoordinator&&) noexcept = default;

  [[nodiscard]] const ModeRegistry& registry() const noexcept {
    return registry_;
  }
  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int boost() const noexcept { return boost_; }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }
  [[nodiscard]] int node_count() const noexcept { return node_count_; }
  [[nodiscard]] std::size_t radial_count() const noexcept {
    return radial_count_;
  }

  // Project arbitrary nodal values onto the coordinator's retained fixed-m
  // harmonic band. This analyze/synthesize path is used to truncate padded
  // nonlinear D/T values before subsequent outer angular derivatives.
  template <class InputView, class OutputView>
  void project(const execution_space& execution, const InputView& input,
               const std::size_t input_field, const OutputView& output,
               const std::size_t output_field) {
    validate_field_views(input, input_field, output, output_field);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto input_mode = Kokkos::subview(input, mode_index, input_field,
                                        Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_angular_projection");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.angular.analyze(execution, input_mode,
                               resource.laplacian_modal);
      resource.angular.synthesize(execution, resource.laplacian_modal,
                                  output_mode);
    }
  }

  // Nodal spin-weighted Laplacian for one field slot. Input and output may be
  // different slots of the same allocation because analysis completes on the
  // same execution queue before synthesis overwrites the output slot.
  template <class InputView, class OutputView>
  void laplacian(const execution_space& execution, const InputView& input,
                 const std::size_t input_field, const OutputView& output,
                 const std::size_t output_field) {
    validate_field_views(input, input_field, output, output_field);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto input_mode = Kokkos::subview(input, mode_index, input_field,
                                        Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_angular_laplacian");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.angular.analyze(execution, input_mode,
                               resource.laplacian_modal);
      resource.angular.laplacian(execution, resource.laplacian_modal,
                                 resource.laplacian_operated);
      resource.angular.synthesize(execution, resource.laplacian_operated,
                                  output_mode);
    }
  }

  template <class FieldView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView>
  void eth(const execution_space& execution, const FieldView& field,
           const std::size_t field_index, const DtView& dt_field,
           const std::size_t dt_field_index, const RadiusView& radius,
           const SinThetaView& sin_theta, const CosThetaView& cos_theta,
           const OutputView& output, const std::size_t output_field) {
    validate_ghp_views(field, field_index, dt_field, dt_field_index, output,
                       output_field, radius, sin_theta, cos_theta);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto field_mode = Kokkos::subview(field, mode_index, field_index,
                                        Kokkos::ALL, Kokkos::ALL);
      auto dt_mode = Kokkos::subview(dt_field, mode_index, dt_field_index,
                                     Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_angular_eth");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.ghp.eth(execution, field_mode, dt_mode, radius, sin_theta,
                       cos_theta, output_mode, resource.ghp_workspace);
    }
  }

  template <class FieldView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView>
  void ethprime(const execution_space& execution, const FieldView& field,
                const std::size_t field_index, const DtView& dt_field,
                const std::size_t dt_field_index, const RadiusView& radius,
                const SinThetaView& sin_theta,
                const CosThetaView& cos_theta, const OutputView& output,
                const std::size_t output_field) {
    validate_ghp_views(field, field_index, dt_field, dt_field_index, output,
                       output_field, radius, sin_theta, cos_theta);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto field_mode = Kokkos::subview(field, mode_index, field_index,
                                        Kokkos::ALL, Kokkos::ALL);
      auto dt_mode = Kokkos::subview(dt_field, mode_index, dt_field_index,
                                     Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_angular_ethprime");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.ghp.ethprime(execution, field_mode, dt_mode, radius, sin_theta,
                            cos_theta, output_mode,
                            resource.ghp_workspace);
    }
  }

  template <class InputView, class OutputView>
  void pure_raise(const execution_space& execution, const InputView& input,
                  const std::size_t input_field, const OutputView& output,
                  const std::size_t output_field) {
    validate_field_views(input, input_field, output, output_field);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto input_mode = Kokkos::subview(input, mode_index, input_field,
                                        Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_pure_raise");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.ghp.pure_raise(execution, input_mode, output_mode,
                              resource.ghp_workspace);
    }
  }

  template <class InputView, class OutputView>
  void pure_lower(const execution_space& execution, const InputView& input,
                  const std::size_t input_field, const OutputView& output,
                  const std::size_t output_field) {
    validate_field_views(input, input_field, output, output_field);
    for (std::size_t mode_index = 0; mode_index < resources_.size();
         ++mode_index) {
      auto input_mode = Kokkos::subview(input, mode_index, input_field,
                                        Kokkos::ALL, Kokkos::ALL);
      auto output_mode = Kokkos::subview(output, mode_index, output_field,
                                         Kokkos::ALL, Kokkos::ALL);
      if (!resources_[mode_index]) {
        zero_mode(execution, output_mode, "zero_inactive_pure_lower");
        continue;
      }
      auto& resource = *resources_[mode_index];
      resource.ghp.pure_lower(execution, input_mode, output_mode,
                              resource.ghp_workspace);
    }
  }

 private:
  template <class ViewType>
  void zero_mode(const execution_space& execution, const ViewType& view,
                 const char* kernel_name) const {
    const std::size_t total = view.extent(0) * view.extent(1);
    Kokkos::parallel_for(
        kernel_name,
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t row = flat / view.extent(1);
          const std::size_t column = flat - row * view.extent(1);
          view(row, column) = Complex(0.0, 0.0);
        });
  }

  template <class ViewType>
  void validate_field_view(const ViewType& view,
                           const std::size_t field_index) const {
    static_assert(ViewType::rank == 4,
                  "coordinator fields must be rank-four Views");
    if (view.extent(0) != registry_.size() ||
        field_index >= view.extent(1) ||
        view.extent(2) != radial_count_ ||
        view.extent(3) != static_cast<std::size_t>(node_count_)) {
      throw std::invalid_argument("angular coordinator field shape mismatch");
    }
  }

  template <class InputView, class OutputView>
  void validate_field_views(const InputView& input,
                            const std::size_t input_field,
                            const OutputView& output,
                            const std::size_t output_field) const {
    validate_field_view(input, input_field);
    validate_field_view(output, output_field);
  }

  template <class FieldView, class DtView, class OutputView,
            class RadiusView, class SinThetaView, class CosThetaView>
  void validate_ghp_views(const FieldView& field,
                          const std::size_t field_index,
                          const DtView& dt_field,
                          const std::size_t dt_field_index,
                          const OutputView& output,
                          const std::size_t output_field,
                          const RadiusView& radius,
                          const SinThetaView& sin_theta,
                          const CosThetaView& cos_theta) const {
    validate_field_view(field, field_index);
    validate_field_view(dt_field, dt_field_index);
    validate_field_view(output, output_field);
    if (radius.extent(0) != radial_count_ ||
        sin_theta.extent(0) != static_cast<std::size_t>(node_count_) ||
        cos_theta.extent(0) != static_cast<std::size_t>(node_count_)) {
      throw std::invalid_argument("angular coordinator coordinate mismatch");
    }
  }

  ModeRegistry registry_;
  int spin_;
  int boost_;
  int ell_max_;
  int node_count_;
  std::size_t radial_count_;
  std::vector<std::unique_ptr<ModeResources>> resources_;
};

}  // namespace teuk
