#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/device_rk4.hpp"
#include "teuk/plus2_checkpoint.hpp"
#include "teuk/plus2_replay.hpp"
#include "teuk/types.hpp"

namespace {

int replay_step_allocations = 0;

void count_replay_step_allocation(
    const Kokkos::Tools::SpaceHandle, const char* name, const void*,
    const std::uint64_t) {
  if (name != nullptr && std::string(name).find("plus2") != std::string::npos) {
    ++replay_step_allocations;
  }
}

teuk::Plus2ReplayConfiguration concurrent_configuration() {
  teuk::Plus2ReplayConfiguration config;
  config.mode = teuk::Plus2RunMode::Concurrent;
  config.ell_max_first = 2;
  config.ell_max_second = 2;
  config.parent_modes = {-1, 1};
  config.target_modes = {-2, 0, 2};
  config.radial_count = 1;
  config.theta_count = 1;
  return config;
}

teuk::Plus2ReplayConfiguration replay_configuration() {
  auto config = concurrent_configuration();
  config.mode = teuk::Plus2RunMode::Replay;
  config.primary_value_count = 2;
  return config;
}

struct Trajectory {
  std::vector<teuk::Complex> primary;
  std::vector<teuk::Complex> companion;
  std::vector<double> primary_times;
  std::vector<double> companion_times;
  std::vector<teuk::SourceActivationState> activations;
};

template <class PrimaryView>
void set_primary(const PrimaryView& primary) {
  auto host = Kokkos::create_mirror_view(primary);
  host(0) = teuk::Complex(0.7, -0.2);
  host(1) = teuk::Complex(-0.4, 0.3);
  Kokkos::deep_copy(primary, host);
}

template <class PrimaryView, class CompanionView>
Trajectory collect(const PrimaryView& primary, const CompanionView& companion,
                   std::vector<double> primary_times,
                   std::vector<double> companion_times,
                   std::vector<teuk::SourceActivationState> activations) {
  const auto primary_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, primary);
  const auto companion_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, companion);
  std::vector<teuk::Complex> primary_values(primary_host.extent(0));
  std::vector<teuk::Complex> companion_values(companion_host.extent(0));
  for (std::size_t i = 0; i < primary_values.size(); ++i) {
    primary_values[i] = primary_host(i);
  }
  for (std::size_t i = 0; i < companion_values.size(); ++i) {
    companion_values[i] = companion_host(i);
  }
  return {std::move(primary_values), std::move(companion_values),
          std::move(primary_times), std::move(companion_times),
          std::move(activations)};
}

Trajectory run_concurrent(const double companion_initial) {
  using range_policy = Kokkos::RangePolicy<teuk::ExecutionSpace>;
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_test_concurrent_primary", 2);
  set_primary(primary);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace>
      primary_workspace(2);
  teuk::Plus2ReplayOrchestrator orchestrator(concurrent_configuration());
  orchestrator.initialize_zero(execution);
  Kokkos::deep_copy(execution, orchestrator.companion_state(),
                    teuk::Complex(companion_initial, 0.0));
  execution.fence("set plus2 companion test initial data");

  std::vector<double> primary_times;
  std::vector<double> companion_times;
  std::vector<teuk::SourceActivationState> activations;
  primary_times.reserve(12);
  companion_times.reserve(12);
  activations.reserve(12);
  const auto primary_rhs = [&](const teuk::ExecutionSpace& stage_execution,
                               const double stage_time, const auto& input,
                               const auto& output) {
    primary_times.push_back(stage_time);
    Kokkos::parallel_for(
        "plus2_test_primary_rhs",
        range_policy(stage_execution, 0, input.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          output(i) = (0.1 + 0.02 * static_cast<double>(i)) * input(i);
        });
  };
  const auto companion_rhs =
      [&](const teuk::ExecutionSpace& stage_execution,
          const double stage_time, const auto& primary_stage,
          const auto& companion_stage, const auto& output,
          const teuk::SourceActivationState activation) {
        companion_times.push_back(stage_time);
        activations.push_back(activation);
        const bool active = activation.active;
        Kokkos::parallel_for(
            "plus2_test_companion_rhs",
            range_policy(stage_execution, 0, companion_stage.extent(0)),
            KOKKOS_LAMBDA(const std::size_t i) {
              const teuk::Complex forcing =
                  active ? (0.3 + 0.01 * static_cast<double>(i)) *
                               (primary_stage(0) + primary_stage(1))
                         : teuk::Complex(0.0, 0.0);
              output(i) = forcing - 0.15 * companion_stage(i);
            });
      };
  const teuk::SourceActivationState activation{true, 0.0, 3, 0.0};
  const auto* stable_pointer = orchestrator.companion_state().data();
  replay_step_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_replay_step_allocation);
  for (int step = 0; step < 3; ++step) {
    orchestrator.advance_concurrent(
        execution, primary, 0.1 * static_cast<double>(step), 0.1, activation,
        primary_rhs, companion_rhs, primary_workspace);
  }
  execution.fence("finish plus2 concurrent test trajectory");
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(replay_step_allocations == 0);
  CHECK(stable_pointer == orchestrator.companion_state().data());
  return collect(primary, orchestrator.companion_state(),
                 std::move(primary_times), std::move(companion_times),
                 std::move(activations));
}

Trajectory run_replay() {
  using range_policy = Kokkos::RangePolicy<teuk::ExecutionSpace>;
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> initial_primary(
      "plus2_test_replay_initial", 2);
  set_primary(initial_primary);
  teuk::Plus2ReplayOrchestrator orchestrator(replay_configuration());
  orchestrator.initialize_zero(execution);
  orchestrator.initialize_replay_primary(execution, initial_primary);

  std::vector<double> primary_times;
  std::vector<double> companion_times;
  std::vector<teuk::SourceActivationState> activations;
  primary_times.reserve(12);
  companion_times.reserve(12);
  activations.reserve(12);
  const auto primary_rhs = [&](const teuk::ExecutionSpace& stage_execution,
                               const double stage_time, const auto& input,
                               const auto& output) {
    primary_times.push_back(stage_time);
    Kokkos::parallel_for(
        "plus2_replay_test_primary_rhs",
        range_policy(stage_execution, 0, input.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          output(i) = (0.1 + 0.02 * static_cast<double>(i)) * input(i);
        });
  };
  const auto companion_rhs =
      [&](const teuk::ExecutionSpace& stage_execution,
          const double stage_time, const auto& primary_stage,
          const auto& companion_stage, const auto& output,
          const teuk::SourceActivationState activation) {
        companion_times.push_back(stage_time);
        activations.push_back(activation);
        const bool active = activation.active;
        Kokkos::parallel_for(
            "plus2_replay_test_companion_rhs",
            range_policy(stage_execution, 0, companion_stage.extent(0)),
            KOKKOS_LAMBDA(const std::size_t i) {
              const teuk::Complex forcing =
                  active ? (0.3 + 0.01 * static_cast<double>(i)) *
                               (primary_stage(0) + primary_stage(1))
                         : teuk::Complex(0.0, 0.0);
              output(i) = forcing - 0.15 * companion_stage(i);
            });
      };
  const teuk::SourceActivationState activation{true, 0.0, 3, 0.0};
  for (int step = 0; step < 3; ++step) {
    orchestrator.advance_replay(
        execution, 0.1 * static_cast<double>(step), 0.1, activation,
        primary_rhs, companion_rhs);
  }
  execution.fence("finish plus2 replay test trajectory");
  return collect(orchestrator.replay_primary_state(),
                 orchestrator.companion_state(), std::move(primary_times),
                 std::move(companion_times), std::move(activations));
}

std::vector<teuk::Complex> run_primary_without_companion() {
  using range_policy = Kokkos::RangePolicy<teuk::ExecutionSpace>;
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_test_disabled_primary", 2);
  set_primary(primary);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  const auto primary_rhs = [&](const teuk::ExecutionSpace& stage_execution,
                               const double, const auto& input,
                               const auto& output) {
    Kokkos::parallel_for(
        "plus2_test_disabled_primary_rhs",
        range_policy(stage_execution, 0, input.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          output(i) = (0.1 + 0.02 * static_cast<double>(i)) * input(i);
        });
  };
  for (int step = 0; step < 3; ++step) {
    teuk::device_classical_rk4_step(
        execution, primary, 0.1 * static_cast<double>(step), 0.1,
        primary_rhs, workspace);
  }
  execution.fence("finish primary trajectory without plus2 companion");
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, primary);
  std::vector<teuk::Complex> result(host.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

class TemporaryCheckpoint {
 public:
  TemporaryCheckpoint() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("teuk-plus2-checkpoint-" + std::to_string(nonce) + ".bin");
  }
  ~TemporaryCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST_CASE("plus2 modes and initial policies are strict and typed") {
  CHECK(teuk::parse_plus2_run_mode("disabled") ==
        teuk::Plus2RunMode::Disabled);
  CHECK(teuk::parse_plus2_run_mode("diagnostic_only") ==
        teuk::Plus2RunMode::DiagnosticOnly);
  CHECK(teuk::parse_plus2_run_mode("concurrent") ==
        teuk::Plus2RunMode::Concurrent);
  CHECK(teuk::parse_plus2_run_mode("replay") == teuk::Plus2RunMode::Replay);
  CHECK(teuk::parse_plus2_initial_policy("zero") ==
        teuk::Plus2InitialPolicy::Zero);
  CHECK(teuk::parse_plus2_initial_policy("checkpoint") ==
        teuk::Plus2InitialPolicy::Checkpoint);

  bool unknown_rejected = false;
  try {
    static_cast<void>(teuk::parse_plus2_run_mode("snapshot"));
  } catch (const std::invalid_argument&) {
    unknown_rejected = true;
  }
  CHECK(unknown_rejected);

  auto invalid = concurrent_configuration();
  invalid.initial_policy = teuk::Plus2InitialPolicy::Checkpoint;
  bool missing_checkpoint_rejected = false;
  try {
    teuk::Plus2ReplayOrchestrator orchestrator(invalid);
    static_cast<void>(orchestrator);
  } catch (const std::invalid_argument&) {
    missing_checkpoint_rejected = true;
  }
  CHECK(missing_checkpoint_rejected);

  teuk::Plus2ReplayConfiguration disabled;
  const teuk::Plus2ReplayOrchestrator disabled_orchestrator(disabled);
  CHECK(!disabled_orchestrator.has_companion());

  teuk::Plus2ReplayConfiguration diagnostic;
  diagnostic.mode = teuk::Plus2RunMode::DiagnosticOnly;
  diagnostic.ell_max_first = 2;
  diagnostic.parent_modes = {-1, 1};
  const teuk::Plus2ReplayOrchestrator diagnostic_orchestrator(diagnostic);
  CHECK(!diagnostic_orchestrator.has_companion());
}

TEST_CASE("plus2 concurrent and deterministic replay are bitwise identical") {
  const auto concurrent = run_concurrent(0.0);
  const auto replay = run_replay();
  CHECK(concurrent.primary == replay.primary);
  CHECK(concurrent.companion == replay.companion);
  CHECK(concurrent.primary_times == replay.primary_times);
  CHECK(concurrent.companion_times == replay.companion_times);
  CHECK(concurrent.primary_times == concurrent.companion_times);
  CHECK(concurrent.primary_times.size() == 12);
  const std::vector<double> first_stage_times{0.0, 0.05, 0.05, 0.1};
  CHECK(std::equal(first_stage_times.begin(), first_stage_times.end(),
                   concurrent.primary_times.begin()));
  CHECK(concurrent.activations.size() == 12);
  for (const auto& activation : concurrent.activations) {
    CHECK(activation.active);
    CHECK(activation.activation_time == 0.0);
    CHECK(activation.consecutive_passes == 3);
    CHECK(activation.last_eligibility_time == 0.0);
  }
}

TEST_CASE("plus2 companion never changes the primary trajectory") {
  const auto negative = run_concurrent(-7.0);
  const auto positive = run_concurrent(11.0);
  const auto disabled = run_primary_without_companion();
  CHECK(negative.primary == positive.primary);
  CHECK(negative.primary == disabled);
  CHECK(negative.companion != positive.companion);
}

TEST_CASE("plus2 checkpoint round trip validates before device mutation") {
  const teuk::ExecutionSpace execution;
  auto storage = teuk::Plus2CompanionStorage::enabled(3, 2, 2,
                                                       "plus2_checkpoint_test");
  auto host = Kokkos::create_mirror_view(storage.flat_state());
  std::vector<teuk::Complex> expected(host.extent(0));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = teuk::Complex(0.125 * static_cast<double>(i + 1),
                                -0.0625 * static_cast<double>(i));
    host(i) = expected[i];
  }
  Kokkos::deep_copy(execution, storage.flat_state(), host);
  execution.fence("set plus2 checkpoint test state");

  teuk::Plus2CheckpointMetadata metadata;
  metadata.parent_modes = {-1, 1};
  metadata.target_modes = {-2, 0, 2};
  metadata.progress = {0.5, 5};
  metadata.source_activation = {true, 0.1, 2, 0.4};
  TemporaryCheckpoint checkpoint;
  const auto saved = teuk::save_plus2_checkpoint(
      execution, checkpoint.path(), storage, metadata);
  CHECK(saved.schema == teuk::plus2_checkpoint_schema);
  CHECK(saved.version == teuk::plus2_checkpoint_format_version);
  CHECK(saved.scaling == teuk::plus2_fixed_tetrad_raw_scaling);
  CHECK(saved.registry_schema == teuk::plus2_signed_mode_registry);
  CHECK(saved.state_checksum != 0);

  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(0.0, 0.0));
  const teuk::Plus2CheckpointExpectations expectations{
      teuk::plus2_fixed_tetrad_raw_scaling,
      teuk::plus2_signed_mode_registry,
      {-1, 1},
      {-2, 0, 2},
      2,
      2};
  const auto loaded = teuk::load_plus2_checkpoint(
      execution, checkpoint.path(), storage, expectations);
  CHECK(loaded.progress.time == 0.5);
  CHECK(loaded.progress.step == 5);
  const auto restored = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(restored(i) == expected[i]);
  }

  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(9.0, -4.0));
  execution.fence("set plus2 checkpoint mutation sentinel");
  {
    std::fstream corrupt(checkpoint.path(), std::ios::binary | std::ios::in |
                                                std::ios::out);
    corrupt.seekg(-1, std::ios::end);
    char final_byte = 0;
    corrupt.read(&final_byte, 1);
    corrupt.clear();
    corrupt.seekp(-1, std::ios::end);
    final_byte ^= static_cast<char>(0x1);
    corrupt.write(&final_byte, 1);
  }
  bool corrupt_rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_checkpoint(
        execution, checkpoint.path(), storage, expectations));
  } catch (const std::runtime_error&) {
    corrupt_rejected = true;
  }
  CHECK(corrupt_rejected);
  const auto unchanged = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < unchanged.extent(0); ++i) {
    CHECK(unchanged(i) == teuk::Complex(9.0, -4.0));
  }
}

TEST_CASE("plus2 step rejects inconsistent accepted activation before RHS") {
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_invalid_activation_primary", 2);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  teuk::Plus2ReplayOrchestrator orchestrator(concurrent_configuration());
  orchestrator.initialize_zero(execution);
  int calls = 0;
  const auto primary_rhs = [&](const auto&, double, const auto&, const auto&) {
    ++calls;
  };
  const auto companion_rhs =
      [&](const auto&, double, const auto&, const auto&, const auto&,
          const auto&) { ++calls; };
  bool rejected = false;
  try {
    orchestrator.advance_concurrent(
        execution, primary, 0.2, 0.1,
        teuk::SourceActivationState{true, 0.3, 1, 0.2}, primary_rhs,
        companion_rhs, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(calls == 0);
}
