#pragma once

#include <Kokkos_Core.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "teuk/full_spatial.hpp"
#include "teuk/pipeline_checkpoint.hpp"
#include "teuk/plus2_replay.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

template <class ExecSpace>
class Plus2LiveSourceComposition;
template <class ExecSpace>
struct Plus2RouteBStageSourceInputs;

class Plus2SourceProvenanceAuthority {
 private:
  explicit constexpr Plus2SourceProvenanceAuthority(const void* identity)
      : identity_(identity) {}

  const void* identity_;

  template <class AnyExecSpace>
  friend class Plus2LiveSourceComposition;
  friend class Plus2CompanionPipeline;
};

namespace plus2_pipeline_detail {

inline bool same_binary64(const double left, const double right) {
  return std::bit_cast<std::uint64_t>(left) ==
         std::bit_cast<std::uint64_t>(right);
}

inline bool same_binary64_vector(const std::vector<double>& left,
                                 const std::vector<double>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!same_binary64(left[index], right[index])) return false;
  }
  return true;
}

inline const char* reduction_name(const ReductionEvolution reduction) {
  switch (reduction) {
    case ReductionEvolution::FreeDamped: return "free_damped";
    case ReductionEvolution::StageConstrained: return "stage_constrained";
  }
  throw std::invalid_argument("unsupported spin +2 reduction evolution");
}

inline Plus2ReplayConfiguration bind_actual_companion_pde(
    Plus2ReplayConfiguration configuration,
    const UniformRadialGrid& radial_grid,
    const TeukolskyParameters& parameters,
    const std::vector<double>& theta_coordinates,
    const ReductionEvolution reduction, const double dissipation_strength,
    const RadialDiscretization discretization) {
  const auto fail = [](const char* field) {
    throw std::invalid_argument(std::string("spin +2 companion ") + field +
                                " does not match replay provenance");
  };
  if (configuration.radial_count != radial_grid.size()) fail("radial extent");
  if (configuration.theta_count != theta_coordinates.size()) {
    fail("theta extent");
  }
  if (configuration.radial_discretization != discretization) {
    fail("radial discretization");
  }
  std::vector<double> actual_radial(radial_grid.size());
  for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
    actual_radial[radial] = radial_grid.coordinate(radial);
  }
  const bool legacy_unspecified =
      configuration.mass == 0.0 && configuration.spin == 0.0 &&
      configuration.compactification_length == 0.0 &&
      configuration.radial_coordinates.empty() &&
      configuration.theta_coordinates.empty() &&
      configuration.reduction_mode.empty() &&
      configuration.reduction_damping == 0.0 &&
      configuration.dissipation == 0.0;
  const bool legacy_complete =
      configuration.mass > 0.0 &&
      configuration.compactification_length > 0.0 &&
      configuration.radial_coordinates.size() == radial_grid.size() &&
      configuration.theta_coordinates.size() == theta_coordinates.size() &&
      !configuration.reduction_mode.empty();
  if (!legacy_unspecified && !legacy_complete) {
    fail("physical provenance bundle");
  }
  if (legacy_complete) {
    if (!same_binary64(configuration.mass, parameters.mass)) fail("mass");
    if (!same_binary64(configuration.spin, parameters.spin)) fail("spin");
    if (!same_binary64(configuration.compactification_length,
                       parameters.compactification_length)) {
      fail("compactification length");
    }
    if (!same_binary64_vector(configuration.radial_coordinates,
                              actual_radial)) {
      fail("radial coordinates");
    }
    if (!same_binary64_vector(configuration.theta_coordinates,
                              theta_coordinates)) {
      fail("theta coordinates");
    }
    if (configuration.reduction_mode != reduction_name(reduction)) {
      fail("reduction mode");
    }
    if (!same_binary64(configuration.reduction_damping,
                       parameters.reduction_damping)) {
      fail("reduction damping");
    }
    if (!same_binary64(configuration.dissipation, dissipation_strength)) {
      fail("dissipation");
    }
  }
  if (configuration.source_normalization != plus2_source_normalization) {
    fail("source normalization");
  }
  configuration.mass = parameters.mass;
  configuration.spin = parameters.spin;
  configuration.compactification_length =
      parameters.compactification_length;
  configuration.radial_coordinates = std::move(actual_radial);
  configuration.theta_coordinates = theta_coordinates;
  configuration.reduction_mode = reduction_name(reduction);
  configuration.reduction_damping = parameters.reduction_damping;
  configuration.dissipation = dissipation_strength;
  plus2_replay_detail::validate_configuration(configuration);
  return configuration;
}

}  // namespace plus2_pipeline_detail

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

template <class ExecSpace, class Provider>
class Plus2BoundStageSourceAdapter {
 public:
  [[nodiscard]] Plus2SourceNormalization source_normalization() const {
    return plus2_source_normalization;
  }

  template <class... Arguments>
  decltype(auto) operator()(Arguments&&... arguments) {
    return composition_->evaluate_bound_routeb_stage(
        std::forward<Arguments>(arguments)..., *provider_);
  }

 private:
  Plus2BoundStageSourceAdapter(
      Plus2LiveSourceComposition<ExecSpace>& composition, Provider& provider)
      : composition_(&composition), provider_(&provider) {}

  [[nodiscard]] const void* authority_identity() const {
    return composition_;
  }

  Plus2LiveSourceComposition<ExecSpace>* composition_;
  Provider* provider_;

  template <class OtherExecSpace>
  friend class Plus2LiveSourceComposition;
  friend class Plus2CompanionPipeline;
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
      : orchestrator_(plus2_pipeline_detail::bind_actual_companion_pde(
            std::move(configuration), radial_grid, parameters,
            theta_coordinates, reduction, dissipation_strength,
            discretization)),
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

  Plus2CheckpointMetadata initialize_checkpoint(const ExecutionSpace&) {
    throw std::logic_error(
        "spin +2 checkpoint restore requires verified primary and source "
        "authority");
  }
  Plus2CheckpointMetadata initialize_checkpoint(
      const ExecutionSpace&,
      const VerifiedPrimaryCheckpointReceipt&,
      const Plus2SourceProvenanceAuthority&) {
    throw std::logic_error(
        "spin +2 checkpoint restore requires the verified primary state view");
  }

  template <class PrimaryStateView>
  Plus2CheckpointMetadata initialize_checkpoint(
      const ExecutionSpace& execution, const PrimaryStateView& primary_state,
      const VerifiedPrimaryCheckpointReceipt& primary_receipt,
      const Plus2SourceProvenanceAuthority source_authority) {
    require_checkpoint_source_authority(source_authority);
    require_primary_state_matches_receipt(execution, primary_state,
                                          primary_receipt);
    require_primary_receipt_compatible(primary_receipt);
    const auto expected = checkpoint_expectations(primary_receipt);
    auto metadata = orchestrator_.initialize_checkpoint_bound(
        execution, expected, Plus2PipelineCheckpointAuthority{0});
    if (orchestrator_.configuration().mode == Plus2RunMode::Replay) {
      orchestrator_.initialize_replay_primary_unchecked(execution,
                                                        primary_state);
    }
    primary_receipt_ = primary_receipt;
    record_qualified_source(source_authority.identity_);
    return metadata;
  }

  [[nodiscard]] Plus2CheckpointExpectations checkpoint_expectations(
      const VerifiedPrimaryCheckpointReceipt& primary_receipt) const {
    return bound_plus2_checkpoint_expectations(orchestrator_.configuration(),
                                               primary_receipt);
  }

  template <class PrimaryStateView>
  Plus2CheckpointMetadata save_checkpoint(
      const ExecutionSpace& execution, const std::filesystem::path& path,
      const Plus2CheckpointProgress progress,
      const SourceActivationState& source_activation,
      const VerifiedPrimaryCheckpointReceipt& primary_receipt,
      const PrimaryStateView& primary_state,
      const Plus2SourceProvenanceAuthority source_authority) const {
    require_checkpoint_source_authority(source_authority);
    if (!orchestrator_.initialized()) {
      throw std::logic_error(
          "spin +2 companion state is not initialized for checkpointing");
    }
    if (primary_receipt.time() != progress.time ||
        primary_receipt.step() != progress.step) {
      throw std::invalid_argument(
          "primary and companion checkpoints must describe the same progress");
    }
    require_primary_state_matches_receipt(execution, primary_state,
                                          primary_receipt);
    require_primary_receipt_compatible(primary_receipt);
    const auto expected = checkpoint_expectations(primary_receipt);
    Plus2CheckpointMetadata metadata;
    metadata.scaling = expected.scaling;
    metadata.registry_schema = expected.registry_schema;
    metadata.parent_modes = expected.parent_modes;
    metadata.target_modes = expected.target_modes;
    metadata.ell_max_first = expected.ell_max_first;
    metadata.ell_max_second = expected.ell_max_second;
    metadata.linear_method = expected.linear_method;
    metadata.second_method = expected.second_method;
    // The artifact is restart input regardless of how the original state was
    // initialized.  The original free-data policy belongs in run provenance;
    // the checkpoint contract records the policy required to consume it.
    metadata.initial_policy = Plus2InitialPolicy::Checkpoint;
    metadata.git_commit = expected.git_commit;
    metadata.runtime_config_schema_version =
        expected.runtime_config_schema_version;
    metadata.radial_count = expected.radial_count;
    metadata.theta_count = expected.theta_count;
    metadata.radial_discretization = expected.radial_discretization;
    metadata.mass = expected.mass;
    metadata.spin = expected.spin;
    metadata.compactification_length = expected.compactification_length;
    metadata.radial_coordinates = expected.radial_coordinates;
    metadata.theta_coordinates = expected.theta_coordinates;
    metadata.time_step = expected.time_step;
    metadata.reduction_mode = expected.reduction_mode;
    metadata.reduction_damping = expected.reduction_damping;
    metadata.dissipation = expected.dissipation;
    metadata.provenance_binding_schema = expected.provenance_binding_schema;
    metadata.source_normalization = expected.source_normalization;
    metadata.primary_checkpoint_identity =
        expected.primary_checkpoint_identity;
    metadata.progress = progress;
    metadata.source_activation = source_activation;
    return save_plus2_pipeline_checkpoint(
        Plus2PipelineCheckpointAuthority{0}, execution, path,
        orchestrator_.companion_storage(), std::move(metadata));
  }

  // Allocation-free common-stage RHS seam used by a higher-level triangular
  // coordinator.  The source adapter may bind additional passive state (for
  // example the same-stage Bianchi adapters), while the primary and companion
  // stage views remain read-only.
  template <class PrimaryStageView, class CompanionStageView,
            class OutputView, class SourceAdapter, class AngularAction>
  void evaluate_common_stage_rhs(
      const ExecutionSpace& execution, const double stage_time,
      const PrimaryStageView& primary_stage,
      const CompanionStageView& companion_stage_flat,
      const OutputView& output_flat,
      const SourceActivationState& activation_snapshot,
      SourceAdapter&& source_adapter, AngularAction&& angular_action) {
    const void* source_identity = require_source_authority(source_adapter);
    require_source_history_compatible(source_identity);
    if (!orchestrator_.initialized()) {
      throw std::logic_error("spin +2 companion state is not initialized");
    }
    plus2_replay_detail::validate_accepted_activation(activation_snapshot,
                                                       stage_time);
    const auto companion_stage = reshape(companion_stage_flat);
    const auto output = reshape(output_flat);
    auto& storage = orchestrator_.companion_storage();
    angular_action(execution, stage_time, companion_stage,
                   storage.angular_laplacian());
    source_adapter(execution, stage_time, primary_stage,
                   Plus2StageSourceTarget{activation_snapshot, forcing_});
    evaluate_sbp_teukolsky_full_stage_rhs(
        execution, radial_grid_, parameters_, theta_, modes_, companion_stage,
        storage.angular_laplacian(), forcing_, reduction_,
        storage.radial_scratch(), output, dissipation_strength_, {}, {},
        discretization_);
    record_qualified_source(source_identity);
  }

  template <class PrimaryStageView, class CompanionStageView,
            class OutputView, class SourceAdapter, class AngularAction>
  void evaluate_common_stage_rhs_validation_only(
      const ExecutionSpace& execution, const double stage_time,
      const PrimaryStageView& primary_stage,
      const CompanionStageView& companion_stage_flat,
      const OutputView& output_flat,
      const SourceActivationState& activation_snapshot,
      SourceAdapter&& source_adapter, AngularAction&& angular_action) {
    record_validation_only_source();
    evaluate_common_stage_rhs_unchecked(
        execution, stage_time, primary_stage, companion_stage_flat, output_flat,
        activation_snapshot, std::forward<SourceAdapter>(source_adapter),
        std::forward<AngularAction>(angular_action));
  }

  template <class PrimaryStateView>
  void initialize_replay_primary(const ExecutionSpace& execution,
                                 const PrimaryStateView& initial_primary,
                                 const VerifiedPrimaryCheckpointReceipt&
                                     primary_receipt) {
    require_primary_state_matches_receipt(execution, initial_primary,
                                          primary_receipt);
    require_primary_receipt_compatible(primary_receipt);
    orchestrator_.initialize_replay_primary_unchecked(execution,
                                                      initial_primary);
    primary_receipt_ = primary_receipt;
  }

  template <class PrimaryStateView>
  void initialize_replay_primary_validation_only(
      const ExecutionSpace& execution,
      const PrimaryStateView& initial_primary) {
    if (orchestrator_.configuration().initial_policy ==
        Plus2InitialPolicy::Checkpoint) {
      throw std::logic_error(
          "checkpoint replay requires a verified primary receipt");
    }
    orchestrator_.initialize_replay_primary_unchecked(execution,
                                                      initial_primary);
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
    const void* source_identity = require_source_authority(source_adapter);
    require_source_history_compatible(source_identity);
    auto companion_rhs = make_companion_rhs(source_adapter, angular_action);
    orchestrator_.advance_concurrent(
        execution, primary, accepted_time, step, accepted_activation,
        std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs,
        primary_workspace);
    record_qualified_source(source_identity);
  }

  template <class PrimaryStateView, class PrimaryRightHandSide,
            class SourceAdapter, class AngularAction>
  void advance_concurrent_validation_only(
      const ExecutionSpace& execution, const PrimaryStateView& primary,
      const double accepted_time, const double step,
      const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs, SourceAdapter&& source_adapter,
      AngularAction&& angular_action,
      DeviceRK4Workspace<Complex, ExecutionSpace>& primary_workspace) {
    record_validation_only_source();
    auto companion_rhs = make_companion_rhs_validation_only(source_adapter,
                                                             angular_action);
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
    const void* source_identity = require_source_authority(source_adapter);
    require_source_history_compatible(source_identity);
    auto companion_rhs = make_companion_rhs(source_adapter, angular_action);
    orchestrator_.advance_replay(
        execution, accepted_time, step, accepted_activation,
        std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs);
    record_qualified_source(source_identity);
  }

  template <class PrimaryRightHandSide, class SourceAdapter,
            class AngularAction>
  void advance_replay_validation_only(
      const ExecutionSpace& execution, const double accepted_time,
      const double step, const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs, SourceAdapter&& source_adapter,
      AngularAction&& angular_action) {
    record_validation_only_source();
    auto companion_rhs = make_companion_rhs_validation_only(source_adapter,
                                                             angular_action);
    orchestrator_.advance_replay(
        execution, accepted_time, step, accepted_activation,
        std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs);
  }

 private:
  template <class ExecSpace, class Provider>
  static const void* require_source_authority(
      const Plus2BoundStageSourceAdapter<ExecSpace, Provider>& source_adapter) {
    if (source_adapter.source_normalization() != plus2_source_normalization ||
        source_adapter.authority_identity() == nullptr) {
      throw std::invalid_argument(
          "spin +2 source adapter normalization is not checkpoint-capable");
    }
    return source_adapter.authority_identity();
  }

  template <class SourceAdapter>
  static const void* require_source_authority(const SourceAdapter&) {
    throw std::invalid_argument(
        "raw spin +2 source callbacks are validation-only and unbound");
  }

  void require_checkpoint_source_authority(
      const Plus2SourceProvenanceAuthority authority) const {
    if (authority.identity_ == nullptr ||
        source_history_ == SourceHistory::ValidationOnly ||
        (source_history_ == SourceHistory::Qualified &&
         source_identity_ != authority.identity_)) {
      throw std::invalid_argument(
          "spin +2 checkpoint source provenance does not match its trajectory");
    }
  }

  void require_source_history_compatible(const void* identity) const {
    if (source_history_ == SourceHistory::ValidationOnly ||
        (source_history_ == SourceHistory::Qualified &&
         source_identity_ != identity)) {
      throw std::invalid_argument(
          "spin +2 source authority changed across the trajectory");
    }
  }

  void record_qualified_source(const void* identity) {
    source_history_ = SourceHistory::Qualified;
    source_identity_ = identity;
  }

  void record_validation_only_source() {
    source_history_ = SourceHistory::ValidationOnly;
    source_identity_ = nullptr;
  }

  void require_primary_receipt_compatible(
      const VerifiedPrimaryCheckpointReceipt& receipt) const {
    if (primary_receipt_ && *primary_receipt_ != receipt) {
      throw std::invalid_argument(
          "spin +2 pipeline primary checkpoint receipt changed");
    }
  }

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
      evaluate_common_stage_rhs(
          stage_execution, stage_time, primary_stage, companion_stage_flat,
          output_flat, activation_snapshot, source_adapter, angular_action);
    };
  }

  template <class PrimaryStageView, class CompanionStageView,
            class OutputView, class SourceAdapter, class AngularAction>
  void evaluate_common_stage_rhs_unchecked(
      const ExecutionSpace& execution, const double stage_time,
      const PrimaryStageView& primary_stage,
      const CompanionStageView& companion_stage_flat,
      const OutputView& output_flat,
      const SourceActivationState& activation_snapshot,
      SourceAdapter&& source_adapter, AngularAction&& angular_action) {
    if (!orchestrator_.initialized()) {
      throw std::logic_error("spin +2 companion state is not initialized");
    }
    plus2_replay_detail::validate_accepted_activation(activation_snapshot,
                                                       stage_time);
    const auto companion_stage = reshape(companion_stage_flat);
    const auto output = reshape(output_flat);
    auto& storage = orchestrator_.companion_storage();
    angular_action(execution, stage_time, companion_stage,
                   storage.angular_laplacian());
    source_adapter(execution, stage_time, primary_stage,
                   Plus2StageSourceTarget{activation_snapshot, forcing_});
    evaluate_sbp_teukolsky_full_stage_rhs(
        execution, radial_grid_, parameters_, theta_, modes_, companion_stage,
        storage.angular_laplacian(), forcing_, reduction_,
        storage.radial_scratch(), output, dissipation_strength_, {}, {},
        discretization_);
  }
  template <class SourceAdapter, class AngularAction>
  auto make_companion_rhs_validation_only(SourceAdapter& source_adapter,
                                          AngularAction& angular_action) {
    return [this, &source_adapter, &angular_action](
               const ExecutionSpace& stage_execution,
               const double stage_time, const auto& primary_stage,
               const auto& companion_stage_flat, const auto& output_flat,
               const SourceActivationState activation_snapshot) {
      if (!orchestrator_.initialized()) {
        throw std::logic_error("spin +2 companion state is not initialized");
      }
      plus2_replay_detail::validate_accepted_activation(activation_snapshot,
                                                         stage_time);
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
  std::optional<VerifiedPrimaryCheckpointReceipt> primary_receipt_;
  enum class SourceHistory { Unbound, Qualified, ValidationOnly };
  SourceHistory source_history_ = SourceHistory::Unbound;
  const void* source_identity_ = nullptr;
};

}  // namespace teuk
