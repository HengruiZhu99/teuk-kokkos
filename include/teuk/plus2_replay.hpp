#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "teuk/device_rk4.hpp"
#include "teuk/plus2_checkpoint.hpp"
#include "teuk/plus2_companion_storage.hpp"
#include "teuk/plus2_runtime_types.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

struct Plus2ReplayConfiguration {
  Plus2RunMode mode = Plus2RunMode::Disabled;
  Plus2InitialPolicy initial_policy = Plus2InitialPolicy::Zero;
  Plus2LinearMethod linear_method = Plus2LinearMethod::MetricCurvature;
  Plus2SecondMethod second_method = Plus2SecondMethod::SourcedCompanion;
  std::filesystem::path checkpoint;
  int ell_max_first = 0;
  int ell_max_second = 0;
  std::vector<int> parent_modes;
  std::vector<int> target_modes;
  std::size_t primary_value_count = 0;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
  std::string git_commit;
  int runtime_config_schema_version = 0;
};

namespace plus2_replay_detail {

inline void require_registry(const std::vector<int>& modes,
                             const int ell_max, const char* label) {
  if (ell_max < 2 || modes.empty() ||
      !std::is_sorted(modes.begin(), modes.end()) ||
      std::adjacent_find(modes.begin(), modes.end()) != modes.end()) {
    throw std::invalid_argument(std::string(label) +
                                " registry is invalid");
  }
  for (const int mode : modes) {
    if (std::abs(mode) > ell_max ||
        !std::binary_search(modes.begin(), modes.end(), -mode)) {
      throw std::invalid_argument(std::string(label) +
                                  " registry must be bandlimited and sharp closed");
    }
  }
}

inline bool evolves_companion(const Plus2RunMode mode) {
  return mode == Plus2RunMode::Concurrent || mode == Plus2RunMode::Replay;
}

inline void validate_configuration(const Plus2ReplayConfiguration& config) {
  (void)plus2_run_mode_name(config.mode);
  (void)plus2_initial_policy_name(config.initial_policy);
  (void)plus2_linear_method_name(config.linear_method);
  (void)plus2_second_method_name(config.second_method);
  if (config.mode == Plus2RunMode::Disabled) {
    if (config.initial_policy != Plus2InitialPolicy::Zero ||
        !config.checkpoint.empty() || config.ell_max_first != 0 ||
        config.ell_max_second != 0 || !config.parent_modes.empty() ||
        !config.target_modes.empty() || config.primary_value_count != 0 ||
        config.radial_count != 0 || config.theta_count != 0) {
      throw std::invalid_argument(
          "disabled plus2 mode must not request storage or registries");
    }
    return;
  }
  require_registry(config.parent_modes, config.ell_max_first, "parent");
  if (config.mode == Plus2RunMode::DiagnosticOnly) {
    if (config.initial_policy != Plus2InitialPolicy::Zero ||
        !config.checkpoint.empty() || config.ell_max_second != 0 ||
        !config.target_modes.empty() || config.primary_value_count != 0 ||
        config.radial_count != 0 || config.theta_count != 0) {
      throw std::invalid_argument(
          "diagnostic-only plus2 mode cannot request a second-order state");
    }
    return;
  }
  require_registry(config.target_modes, config.ell_max_second, "target");
  if (config.radial_count == 0 || config.theta_count == 0) {
    throw std::invalid_argument("plus2 companion extents must be nonzero");
  }
  if (config.mode == Plus2RunMode::Replay && config.primary_value_count == 0) {
    throw std::invalid_argument("plus2 replay needs a primary state extent");
  }
  if (config.mode == Plus2RunMode::Concurrent &&
      config.primary_value_count != 0) {
    throw std::invalid_argument(
        "concurrent plus2 mode uses caller-owned primary state");
  }
  if ((config.initial_policy == Plus2InitialPolicy::Checkpoint) !=
      !config.checkpoint.empty()) {
    throw std::invalid_argument(
        "plus2 checkpoint policy and path must be specified together");
  }
  plus2_checkpoint_detail::require_git_commit(config.git_commit);
  if (config.runtime_config_schema_version <= 0) {
    throw std::invalid_argument(
        "plus2 evolving mode requires a runtime config schema version");
  }
}

inline void validate_accepted_activation(const SourceActivationState& state,
                                         const double accepted_time) {
  if (!std::isfinite(accepted_time) || accepted_time < 0.0 ||
      !std::isfinite(state.activation_time) ||
      !std::isfinite(state.last_eligibility_time) ||
      state.consecutive_passes < 0 ||
      (state.active &&
       (state.activation_time < 0.0 || state.activation_time > accepted_time)) ||
      (!state.active && state.activation_time != -1.0) ||
      state.last_eligibility_time > accepted_time) {
    throw std::invalid_argument(
        "invalid accepted source-activation state for plus2 step");
  }
}

}  // namespace plus2_replay_detail

inline Plus2CheckpointExpectations plus2_checkpoint_expectations(
    const Plus2ReplayConfiguration& configuration) {
  plus2_replay_detail::validate_configuration(configuration);
  if (!plus2_replay_detail::evolves_companion(configuration.mode)) {
    throw std::invalid_argument(
        "only an evolving plus2 mode has checkpoint expectations");
  }
  Plus2CheckpointExpectations expected;
  expected.parent_modes = configuration.parent_modes;
  expected.target_modes = configuration.target_modes;
  expected.ell_max_first = configuration.ell_max_first;
  expected.ell_max_second = configuration.ell_max_second;
  expected.linear_method = configuration.linear_method;
  expected.second_method = configuration.second_method;
  expected.initial_policy = configuration.initial_policy;
  expected.git_commit = configuration.git_commit;
  expected.runtime_config_schema_version =
      configuration.runtime_config_schema_version;
  expected.radial_count = configuration.radial_count;
  expected.theta_count = configuration.theta_count;
  return expected;
}

// Standalone orchestration slice for the passive spin +2 state. It is not a
// SpatialPipeline member and contains no curvature or source formula. The
// companion callback receives an immutable copy of the primary pipeline's
// accepted-state activation latch at every common RK stage.
class Plus2ReplayOrchestrator {
 public:
  explicit Plus2ReplayOrchestrator(Plus2ReplayConfiguration configuration)
      : configuration_(std::move(configuration)) {
    plus2_replay_detail::validate_configuration(configuration_);
    if (plus2_replay_detail::evolves_companion(configuration_.mode)) {
      companion_ = Plus2CompanionStorage::enabled(
          configuration_.target_modes.size(), configuration_.radial_count,
          configuration_.theta_count, "plus2_orchestrated");
    }
    if (configuration_.mode == Plus2RunMode::Replay) {
      replay_primary_.emplace("plus2_replay_primary",
                              configuration_.primary_value_count);
      replay_primary_workspace_.emplace(configuration_.primary_value_count);
    }
  }

  [[nodiscard]] const Plus2ReplayConfiguration& configuration() const {
    return configuration_;
  }
  [[nodiscard]] bool has_companion() const {
    return companion_.is_enabled();
  }
  [[nodiscard]] bool initialized() const { return companion_initialized_; }
  [[nodiscard]] Plus2CompanionStorage& companion_storage() {
    require_companion();
    return companion_;
  }
  [[nodiscard]] const Plus2CompanionStorage& companion_storage() const {
    require_companion();
    return companion_;
  }
  [[nodiscard]] Plus2CompanionFlatView companion_state() const {
    require_companion();
    return companion_.flat_state();
  }
  [[nodiscard]] Kokkos::View<Complex*, MemorySpace> replay_primary_state() const {
    if (!replay_primary_) {
      throw std::logic_error("plus2 orchestrator is not in replay mode");
    }
    return *replay_primary_;
  }

  void initialize_zero(const ExecutionSpace& execution) {
    require_companion();
    if (configuration_.initial_policy != Plus2InitialPolicy::Zero) {
      throw std::logic_error("plus2 run requires checkpoint initialization");
    }
    Kokkos::deep_copy(execution, companion_.flat_state(), Complex(0.0, 0.0));
    execution.fence("initialize zero plus2 companion");
    companion_initialized_ = true;
  }

  Plus2CheckpointMetadata initialize_checkpoint(
      const ExecutionSpace& execution) {
    require_companion();
    if (configuration_.initial_policy != Plus2InitialPolicy::Checkpoint) {
      throw std::logic_error("plus2 run does not request a checkpoint");
    }
    const auto expected = plus2_checkpoint_expectations(configuration_);
    auto metadata = load_plus2_checkpoint(
        execution, configuration_.checkpoint, companion_, expected);
    companion_initialized_ = true;
    return metadata;
  }

  template <class PrimaryStateView>
  void initialize_replay_primary(const ExecutionSpace& execution,
                                 const PrimaryStateView& initial_primary) {
    static_assert(PrimaryStateView::rank == 1,
                  "replay primary state must be rank one");
    if (!replay_primary_ ||
        initial_primary.extent(0) != configuration_.primary_value_count) {
      throw std::invalid_argument("replay primary state extent mismatch");
    }
    Kokkos::deep_copy(execution, *replay_primary_, initial_primary);
    execution.fence("initialize deterministic plus2 replay primary state");
    replay_primary_initialized_ = true;
  }

  template <class PrimaryStateView, class PrimaryRightHandSide,
            class CompanionRightHandSide>
  void advance_concurrent(
      const ExecutionSpace& execution, const PrimaryStateView& primary,
      const double accepted_time, const double step,
      const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs,
      CompanionRightHandSide&& companion_rhs,
      DeviceRK4Workspace<Complex, ExecutionSpace>& primary_workspace) {
    if (configuration_.mode != Plus2RunMode::Concurrent) {
      throw std::logic_error("plus2 orchestrator is not in concurrent mode");
    }
    advance(execution, primary, accepted_time, step, accepted_activation,
            std::forward<PrimaryRightHandSide>(primary_rhs),
            std::forward<CompanionRightHandSide>(companion_rhs),
            primary_workspace);
  }

  template <class PrimaryRightHandSide, class CompanionRightHandSide>
  void advance_replay(
      const ExecutionSpace& execution, const double accepted_time,
      const double step, const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs,
      CompanionRightHandSide&& companion_rhs) {
    if (configuration_.mode != Plus2RunMode::Replay || !replay_primary_ ||
        !replay_primary_workspace_ || !replay_primary_initialized_) {
      throw std::logic_error("plus2 replay primary state is not initialized");
    }
    advance(execution, *replay_primary_, accepted_time, step,
            accepted_activation,
            std::forward<PrimaryRightHandSide>(primary_rhs),
            std::forward<CompanionRightHandSide>(companion_rhs),
            *replay_primary_workspace_);
  }

 private:
  void require_companion() const {
    if (!companion_.is_enabled()) {
      throw std::logic_error("plus2 mode has no second-order companion state");
    }
  }

  template <class PrimaryStateView, class PrimaryRightHandSide,
            class CompanionRightHandSide>
  void advance(
      const ExecutionSpace& execution, const PrimaryStateView& primary,
      const double accepted_time, const double step,
      const SourceActivationState& accepted_activation,
      PrimaryRightHandSide&& primary_rhs,
      CompanionRightHandSide&& companion_rhs,
      DeviceRK4Workspace<Complex, ExecutionSpace>& primary_workspace) {
    require_companion();
    if (!companion_initialized_) {
      throw std::logic_error("plus2 companion state is not initialized");
    }
    if (!std::isfinite(step) || !(step > 0.0)) {
      throw std::invalid_argument("plus2 step must be positive and finite");
    }
    plus2_replay_detail::validate_accepted_activation(accepted_activation,
                                                       accepted_time);
    const SourceActivationState activation_snapshot = accepted_activation;
    auto stage_companion_rhs =
        [&companion_rhs, activation_snapshot](
            const ExecutionSpace& stage_execution, const double stage_time,
            const auto& primary_stage, const auto& companion_stage,
            const auto& output) {
          companion_rhs(stage_execution, stage_time, primary_stage,
                        companion_stage, output, activation_snapshot);
        };
    device_one_way_coupled_rk4_step(
        execution, primary, companion_.flat_state(), accepted_time, step,
        std::forward<PrimaryRightHandSide>(primary_rhs), stage_companion_rhs,
        primary_workspace, companion_.rk_workspace());
  }

  Plus2ReplayConfiguration configuration_;
  Plus2CompanionStorage companion_;
  std::optional<Kokkos::View<Complex*, MemorySpace>> replay_primary_;
  std::optional<DeviceRK4Workspace<Complex, ExecutionSpace>>
      replay_primary_workspace_;
  bool companion_initialized_ = false;
  bool replay_primary_initialized_ = false;
};

}  // namespace teuk
