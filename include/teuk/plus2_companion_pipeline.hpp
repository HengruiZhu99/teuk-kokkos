#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "teuk/full_spatial.hpp"
#include "teuk/plus2_replay.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

using Plus2CompanionForcingView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using Plus2CompanionModeView = Kokkos::View<int*, MemorySpace>;
using Plus2CompanionThetaView = Kokkos::View<Real*, MemorySpace>;
using UnmanagedPlus2CompanionFieldView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Typed destination handed to a stage-source adapter.  The activation value
// is the immutable accepted-step snapshot captured by Plus2ReplayOrchestrator,
// not a latch recomputed at an intermediate RK stage.  The adapter must fill
// every entry of coordinate_forcing on the supplied execution-space instance.
// In particular, it must write zero when accepted_activation.active is false.
// Plus2SourceSpatialWorkspace::forcing_value() has this same logical
// (mode,radial,theta) representation and can be copied or gathered here.
struct Plus2StageSourceTarget {
  SourceActivationState accepted_activation;
  Plus2CompanionForcingView coordinate_forcing;
};

// Standalone, one-way spin +2 companion integration slice.  It deliberately
// does not construct any of the fourteen qualified source primitives and is
// not connected to solver_driver.  A source adapter supplies the already
// derived coordinate forcing from the primary stage; an angular adapter
// supplies the spin +2 angular Laplacian of the companion Psi field.
//
// SourceAdapter signature:
//   source(exec, stage_time, primary_stage, Plus2StageSourceTarget)
// AngularAction signature:
//   angular(exec, stage_time, companion_stage, angular_laplacian)
//
// Both adapters must enqueue work without allocation, host/device transfer,
// or a fence.  The common-stage RK4 and replay/checkpoint ownership are reused
// from Plus2ReplayOrchestrator rather than duplicated here.
class Plus2CompanionPipeline {
 public:
  Plus2CompanionPipeline(Plus2ReplayConfiguration configuration,
                         const UniformRadialGrid& radial_grid,
                         const TeukolskyParameters& parameters,
                         std::vector<double> theta_coordinates,
                         const ReductionEvolution reduction,
                         const double dissipation_strength = 0.0,
                         const std::string& label = "plus2_pipeline",
                         const RadialDiscretization discretization =
                             RadialDiscretization::D42)
      : orchestrator_(std::move(configuration)),
        radial_grid_(radial_grid),
        parameters_(parameters),
        reduction_(reduction),
        dissipation_strength_(dissipation_strength),
        discretization_(discretization),
        modes_(label + "_modes",
               orchestrator_.configuration().target_modes.size()),
        theta_(label + "_theta", theta_coordinates.size()),
        forcing_(label + "_forcing",
                 orchestrator_.configuration().target_modes.size(),
                 radial_grid.size(), theta_coordinates.size()) {
    const auto& config = orchestrator_.configuration();
    if (!plus2_replay_detail::evolves_companion(config.mode)) {
      throw std::invalid_argument(
          "spin +2 companion pipeline requires concurrent or replay mode");
    }
    if (parameters_.spin_weight != 2) {
      throw std::invalid_argument(
          "spin +2 companion pipeline requires spin_weight=+2");
    }
    if (config.radial_count != radial_grid_.size() ||
        config.theta_count != theta_coordinates.size()) {
      throw std::invalid_argument(
          "spin +2 companion pipeline grid extents do not match config");
    }
    if (config.radial_discretization != discretization_) {
      throw std::invalid_argument(
          "spin +2 companion radial discretization does not match config");
    }
    if (radial_grid_.size() < radial_minimum_points(discretization_)) {
      throw std::invalid_argument(
          "spin +2 companion grid is too short for its radial stencil");
    }
    if (dissipation_strength_ < 0.0 ||
        !std::isfinite(dissipation_strength_)) {
      throw std::invalid_argument(
          "spin +2 companion dissipation must be finite and nonnegative");
    }
    for (const double angle : theta_coordinates) {
      if (!std::isfinite(angle) || !(angle > 0.0) ||
          !(angle < 3.141592653589793238462643383279502884)) {
        throw std::invalid_argument(
            "spin +2 companion theta coordinates must be interior and finite");
      }
    }

    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_theta = Kokkos::create_mirror_view(theta_);
    for (std::size_t i = 0; i < config.target_modes.size(); ++i) {
      host_modes(i) = config.target_modes[i];
    }
    for (std::size_t i = 0; i < theta_coordinates.size(); ++i) {
      host_theta(i) = theta_coordinates[i];
    }
    Kokkos::deep_copy(modes_, host_modes);
    Kokkos::deep_copy(theta_, host_theta);
  }

  Plus2CompanionPipeline(const Plus2CompanionPipeline&) = delete;
  Plus2CompanionPipeline& operator=(const Plus2CompanionPipeline&) = delete;
  Plus2CompanionPipeline(Plus2CompanionPipeline&&) noexcept = default;
  Plus2CompanionPipeline& operator=(Plus2CompanionPipeline&&) noexcept =
      default;

  [[nodiscard]] const Plus2ReplayConfiguration& configuration() const {
    return orchestrator_.configuration();
  }
  [[nodiscard]] Plus2ReplayOrchestrator& orchestrator() {
    return orchestrator_;
  }
  [[nodiscard]] const Plus2ReplayOrchestrator& orchestrator() const {
    return orchestrator_;
  }
  [[nodiscard]] Plus2CompanionStorage& companion_storage() {
    return orchestrator_.companion_storage();
  }
  [[nodiscard]] const Plus2CompanionStorage& companion_storage() const {
    return orchestrator_.companion_storage();
  }
  [[nodiscard]] Plus2CompanionFlatView companion_state() const {
    return orchestrator_.companion_state();
  }
  [[nodiscard]] Plus2CompanionModeView modes() const { return modes_; }
  [[nodiscard]] Plus2CompanionThetaView theta() const { return theta_; }
  [[nodiscard]] Plus2CompanionForcingView forcing() const { return forcing_; }

  void initialize_zero(const ExecutionSpace& execution) {
    orchestrator_.initialize_zero(execution);
  }

  Plus2CheckpointMetadata initialize_checkpoint(
      const ExecutionSpace& execution) {
    return orchestrator_.initialize_checkpoint(execution);
  }

  template <class PrimaryStateView>
  void initialize_replay_primary(const ExecutionSpace& execution,
                                 const PrimaryStateView& initial_primary) {
    orchestrator_.initialize_replay_primary(execution, initial_primary);
  }

  template <class PrimaryStateView, class PrimaryRightHandSide,
            class SourceAdapter, class AngularAction>
  void advance_concurrent(
      const ExecutionSpace& execution, const PrimaryStateView& primary,
      const double accepted_time, const double step,
      const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs, SourceAdapter&& source_adapter,
      AngularAction&& angular_action,
      DeviceRK4Workspace<Complex, ExecutionSpace>& primary_workspace) {
    auto companion_rhs = make_companion_rhs(source_adapter, angular_action);
    orchestrator_.advance_concurrent(
        execution, primary, accepted_time, step, accepted_activation,
        std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs,
        primary_workspace);
  }

  template <class PrimaryRightHandSide, class SourceAdapter,
            class AngularAction>
  void advance_replay(
      const ExecutionSpace& execution, const double accepted_time,
      const double step, const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs, SourceAdapter&& source_adapter,
      AngularAction&& angular_action) {
    auto companion_rhs = make_companion_rhs(source_adapter, angular_action);
    orchestrator_.advance_replay(
        execution, accepted_time, step, accepted_activation,
        std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs);
  }

 private:
  template <class FlatView>
  [[nodiscard]] UnmanagedPlus2CompanionFieldView reshape(
      const FlatView& flat) const {
    static_assert(FlatView::rank == 1,
                  "spin +2 companion RK stage must be flat");
    const auto& config = orchestrator_.configuration();
    if (flat.extent(0) != orchestrator_.companion_storage().value_count()) {
      throw std::invalid_argument(
          "spin +2 companion RK stage extent does not match storage");
    }
    return UnmanagedPlus2CompanionFieldView(
        flat.data(), config.target_modes.size(),
        static_cast<std::size_t>(TeukolskyField::Count), config.radial_count,
        config.theta_count);
  }

  template <class SourceAdapter, class AngularAction>
  auto make_companion_rhs(SourceAdapter& source_adapter,
                          AngularAction& angular_action) {
    return [this, &source_adapter, &angular_action](
               const ExecutionSpace& stage_execution,
               const double stage_time, const auto& primary_stage,
               const auto& companion_stage_flat, const auto& output_flat,
               const SourceActivationState activation_snapshot) {
      const auto companion_stage = reshape(companion_stage_flat);
      const auto output = reshape(output_flat);
      auto& storage = orchestrator_.companion_storage();
      angular_action(stage_execution, stage_time, companion_stage,
                     storage.angular_laplacian());
      source_adapter(stage_execution, stage_time, primary_stage,
                     Plus2StageSourceTarget{activation_snapshot, forcing_});
      evaluate_sbp_teukolsky_full_stage_rhs(
          stage_execution, radial_grid_, parameters_, theta_, modes_,
          companion_stage, storage.angular_laplacian(), forcing_, reduction_,
          storage.radial_scratch(), output, dissipation_strength_, {}, {},
          discretization_);
    };
  }

  Plus2ReplayOrchestrator orchestrator_;
  UniformRadialGrid radial_grid_;
  TeukolskyParameters parameters_;
  ReductionEvolution reduction_;
  double dissipation_strength_;
  RadialDiscretization discretization_;
  Plus2CompanionModeView modes_;
  Plus2CompanionThetaView theta_;
  Plus2CompanionForcingView forcing_;
};

}  // namespace teuk
