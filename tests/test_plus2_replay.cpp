#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/device_rk4.hpp"
#include "teuk/plus2_checkpoint.hpp"
#include "teuk/plus2_replay.hpp"
#include "teuk/types.hpp"

namespace {

teuk::PrimaryCheckpointContentIdentity test_primary_identity() {
  return teuk::primary_checkpoint_identity_from_hex(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

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
  config.git_commit = teuk::plus2_build_git_commit();
  config.runtime_config_schema_version = 1;
  config.mass = 1.0;
  config.spin = 0.7;
  config.compactification_length = 1.6;
  config.radial_coordinates = {0.0};
  config.theta_coordinates = {0.25};
  config.time_step = 0.1;
  config.reduction_mode = "free_damped";
  config.reduction_damping = 0.1;
  config.dissipation = 0.0;
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
  orchestrator.initialize_replay_primary_validation_only(execution,
                                                         initial_primary);

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

void replace_checkpoint_bytes(const std::filesystem::path& path,
                              const std::string& original,
                              const std::string& replacement) {
  if (original.size() != replacement.size()) {
    throw std::invalid_argument("checkpoint replacement must preserve size");
  }
  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  const auto match = std::search(bytes.begin(), bytes.end(), original.begin(),
                                 original.end());
  if (match == bytes.end()) {
    throw std::runtime_error("checkpoint marker was not found");
  }
  std::copy(replacement.begin(), replacement.end(), match);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("failed to corrupt checkpoint marker");
}

void convert_plus2_checkpoint_to_legacy_v2(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  if (input.bad()) throw std::runtime_error("failed reading checkpoint");
  const std::uint32_t current = teuk::plus2_checkpoint_format_version;
  const auto version_bytes = std::bit_cast<std::array<char, 4>>(current);
  const auto match = std::search(bytes.begin(), bytes.end(),
                                 version_bytes.begin(), version_bytes.end());
  if (match == bytes.end()) throw std::runtime_error("version not found");
  const std::uint32_t legacy =
      teuk::plus2_checkpoint_legacy_representation_version;
  const auto legacy_bytes = std::bit_cast<std::array<char, 4>>(legacy);
  std::copy(legacy_bytes.begin(), legacy_bytes.end(), match);

  const std::string scheme = "d4-2";
  const auto length_bytes = std::bit_cast<std::array<char, 8>>(
      static_cast<std::uint64_t>(scheme.size()));
  const std::vector<char> marker = [&] {
    std::vector<char> result(length_bytes.begin(), length_bytes.end());
    result.insert(result.end(), scheme.begin(), scheme.end());
    return result;
  }();
  const auto scheme_match = std::search(bytes.begin(), bytes.end(),
                                        marker.begin(), marker.end());
  if (scheme_match == bytes.end()) throw std::runtime_error("scheme not found");
  // Retain the v2 radial-scheme field and erase exactly the current v4
  // physical-problem and typed-binding block following it.
  const auto provenance_begin = scheme_match + marker.size();
  const auto string_size = [](const std::string& value) {
    return sizeof(std::uint64_t) + value.size();
  };
  const std::size_t provenance_size =
      3 * sizeof(double) +
      (sizeof(std::uint64_t) + 2 * sizeof(double)) * 2 + sizeof(double) +
      string_size("free_damped") + 2 * sizeof(double) +
      string_size(teuk::plus2_unbound_codec_schema) +
      sizeof(std::uint32_t) +
      string_size(teuk::plus2_source_normalization_name) +
      string_size(teuk::checkpoint_content_hash_algorithm) +
      string_size(teuk::primary_checkpoint_state_schema) +
      string_size(std::string(64, '0'));
  bytes.erase(provenance_begin, provenance_begin + provenance_size);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("failed writing legacy checkpoint");
}

teuk::Plus2CheckpointMetadata checkpoint_metadata() {
  teuk::Plus2CheckpointMetadata metadata;
  metadata.parent_modes = {-1, 1};
  metadata.target_modes = {-2, 0, 2};
  metadata.ell_max_first = 2;
  metadata.ell_max_second = 2;
  metadata.linear_method = teuk::Plus2LinearMethod::MetricCurvature;
  metadata.second_method = teuk::Plus2SecondMethod::SourcedCompanion;
  metadata.initial_policy = teuk::Plus2InitialPolicy::Zero;
  metadata.git_commit = teuk::plus2_build_git_commit();
  metadata.runtime_config_schema_version = 1;
  metadata.radial_discretization = teuk::RadialDiscretization::D42;
  metadata.progress = {0.5, 5};
  metadata.source_activation = {true, 0.1, 2, 0.4};
  metadata.mass = 1.0;
  metadata.spin = 0.7;
  metadata.compactification_length = 1.6;
  metadata.radial_coordinates = {0.0, 0.5};
  metadata.theta_coordinates = {-0.4, 0.4};
  metadata.time_step = 0.1;
  metadata.reduction_mode = "free_damped";
  metadata.reduction_damping = 0.1;
  metadata.dissipation = 0.005;
  metadata.primary_checkpoint_identity = test_primary_identity();
  return metadata;
}

teuk::Plus2CheckpointExpectations checkpoint_expectations() {
  teuk::Plus2CheckpointExpectations expected;
  expected.parent_modes = {-1, 1};
  expected.target_modes = {-2, 0, 2};
  expected.ell_max_first = 2;
  expected.ell_max_second = 2;
  expected.linear_method = teuk::Plus2LinearMethod::MetricCurvature;
  expected.second_method = teuk::Plus2SecondMethod::SourcedCompanion;
  expected.initial_policy = teuk::Plus2InitialPolicy::Zero;
  expected.git_commit = teuk::plus2_build_git_commit();
  expected.runtime_config_schema_version = 1;
  expected.radial_count = 2;
  expected.theta_count = 2;
  expected.radial_discretization = teuk::RadialDiscretization::D42;
  expected.mass = 1.0;
  expected.spin = 0.7;
  expected.compactification_length = 1.6;
  expected.radial_coordinates = {0.0, 0.5};
  expected.theta_coordinates = {-0.4, 0.4};
  expected.time_step = 0.1;
  expected.reduction_mode = "free_damped";
  expected.reduction_damping = 0.1;
  expected.dissipation = 0.005;
  expected.primary_checkpoint_identity = test_primary_identity();
  expected.progress = {0.5, 5};
  return expected;
}

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

  bool unbound_expectations_rejected = false;
  try {
    static_cast<void>(
        teuk::plus2_checkpoint_expectations(replay_configuration()));
  } catch (const std::logic_error&) {
    unbound_expectations_rejected = true;
  }
  CHECK(unbound_expectations_rejected);
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

  auto metadata = checkpoint_metadata();
  metadata.radial_discretization = teuk::RadialDiscretization::D105;
  TemporaryCheckpoint checkpoint;
  const auto saved = teuk::save_plus2_checkpoint(
      execution, checkpoint.path(), storage, metadata);
  CHECK(saved.schema == teuk::plus2_checkpoint_schema);
  CHECK(saved.version == teuk::plus2_checkpoint_format_version);
  CHECK(saved.byte_order == teuk::plus2_native_byte_order());
  CHECK(saved.floating_point_format == teuk::plus2_binary64_format);
  CHECK(saved.complex_component_order ==
        teuk::plus2_complex_component_order);
  CHECK(saved.state_storage_order == teuk::plus2_state_storage_order);
  CHECK(saved.scaling == teuk::plus2_fixed_tetrad_raw_scaling);
  CHECK(saved.scaling.find("Z0_source=Psi0_raw_fixed_tetrad/R^5") !=
        std::string::npos);
  CHECK(saved.registry_schema == teuk::plus2_signed_mode_registry);
  CHECK(saved.ell_max_first == 2);
  CHECK(saved.ell_max_second == 2);
  CHECK(saved.linear_method == teuk::Plus2LinearMethod::MetricCurvature);
  CHECK(saved.second_method == teuk::Plus2SecondMethod::SourcedCompanion);
  CHECK(saved.initial_policy == teuk::Plus2InitialPolicy::Zero);
  CHECK(saved.git_commit == teuk::plus2_build_git_commit());
  CHECK(saved.runtime_config_schema_version == 1);
  CHECK(saved.mass == 1.0);
  CHECK(saved.spin == 0.7);
  CHECK(saved.compactification_length == 1.6);
  CHECK(saved.radial_coordinates == std::vector<double>({0.0, 0.5}));
  CHECK(saved.theta_coordinates == std::vector<double>({-0.4, 0.4}));
  CHECK(saved.time_step == 0.1);
  CHECK(saved.reduction_mode == "free_damped");
  CHECK(saved.reduction_damping == 0.1);
  CHECK(saved.dissipation == 0.005);
  CHECK(saved.provenance_binding_schema ==
        teuk::plus2_unbound_codec_schema);
  CHECK(saved.source_normalization == teuk::plus2_source_normalization);
  CHECK(saved.primary_checkpoint_identity == test_primary_identity());
  CHECK(saved.state_checksum != 0);

  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(0.0, 0.0));
  auto expectations = checkpoint_expectations();
  expectations.radial_discretization = teuk::RadialDiscretization::D105;
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
  auto mismatched_expectations = expectations;
  mismatched_expectations.scaling = "a-different-plus2-scaling";
  bool mismatch_rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_checkpoint(
        execution, checkpoint.path(), storage, mismatched_expectations));
  } catch (const std::runtime_error&) {
    mismatch_rejected = true;
  }
  CHECK(mismatch_rejected);
  const auto unchanged_after_mismatch = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < unchanged_after_mismatch.extent(0); ++i) {
    CHECK(unchanged_after_mismatch(i) == teuk::Complex(9.0, -4.0));
  }

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

TEST_CASE("plus2 checkpoint rejects every scientific metadata mismatch") {
  const teuk::ExecutionSpace execution;
  auto storage = teuk::Plus2CompanionStorage::enabled(
      3, 2, 2, "plus2_scientific_checkpoint_test");
  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(1.0, -2.0));
  TemporaryCheckpoint checkpoint;
  static_cast<void>(teuk::save_plus2_checkpoint(
      execution, checkpoint.path(), storage, checkpoint_metadata()));
  const auto baseline = checkpoint_expectations();

  using Modifier =
      std::function<void(teuk::Plus2CheckpointExpectations&)>;
  const std::vector<Modifier> mismatches{
      [](auto& expected) { expected.ell_max_first = 3; },
      [](auto& expected) { expected.ell_max_second = 3; },
      [](auto& expected) {
        expected.linear_method = teuk::Plus2LinearMethod::Tsi;
      },
      [](auto& expected) {
        expected.second_method = static_cast<teuk::Plus2SecondMethod>(99);
      },
      [](auto& expected) {
        expected.initial_policy = teuk::Plus2InitialPolicy::Checkpoint;
      },
      [](auto& expected) {
        expected.git_commit[0] =
            expected.git_commit[0] == '0' ? '1' : '0';
      },
      [](auto& expected) { ++expected.runtime_config_schema_version; },
      [](auto& expected) {
        expected.radial_discretization = teuk::RadialDiscretization::D84;
      },
      [](auto& expected) { expected.mass = 1.1; },
      [](auto& expected) { expected.spin = -0.7; },
      [](auto& expected) { expected.compactification_length = 1.7; },
      [](auto& expected) { expected.radial_coordinates[1] = 0.51; },
      [](auto& expected) { expected.theta_coordinates[0] = -0.41; },
      [](auto& expected) { expected.time_step = 0.05; },
      [](auto& expected) { expected.reduction_mode = "stage_constrained"; },
      [](auto& expected) { expected.reduction_damping = 0.2; },
      [](auto& expected) { expected.dissipation = 0.006; },
      [](auto& expected) { expected.provenance_binding_schema += "-wrong"; },
      [](auto& expected) {
        expected.source_normalization =
            static_cast<teuk::Plus2SourceNormalization>(99);
      },
      [](auto& expected) {
        expected.primary_checkpoint_identity.digest[7] ^= 0x01U;
      },
      [](auto& expected) {
        expected.primary_checkpoint_identity.state_schema += "-wrong";
      },
      [](auto& expected) {
        expected.primary_checkpoint_identity.algorithm = "fnv1a64";
      }};

  for (const auto& modify : mismatches) {
    auto expected = baseline;
    modify(expected);
    Kokkos::deep_copy(execution, storage.flat_state(),
                      teuk::Complex(-6.0, 4.0));
    execution.fence("set scientific-mismatch mutation sentinel");
    bool rejected = false;
    try {
      static_cast<void>(teuk::load_plus2_checkpoint(
          execution, checkpoint.path(), storage, expected));
    } catch (const std::exception&) {
      rejected = true;
    }
    CHECK(rejected);
    const auto unchanged = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, storage.flat_state());
    for (std::size_t i = 0; i < unchanged.extent(0); ++i) {
      CHECK(unchanged(i) == teuk::Complex(-6.0, 4.0));
    }
  }
}

TEST_CASE("plus2 checkpoint rejects nonfinite provenance before mutation") {
  const teuk::ExecutionSpace execution;
  auto storage = teuk::Plus2CompanionStorage::enabled(
      3, 2, 2, "plus2_nonfinite_checkpoint_test");
  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(7.0, -3.0));
  TemporaryCheckpoint checkpoint;
  auto metadata = checkpoint_metadata();
  metadata.mass = std::numeric_limits<double>::quiet_NaN();
  bool rejected = false;
  try {
    static_cast<void>(teuk::save_plus2_checkpoint(
        execution, checkpoint.path(), storage, metadata));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(!std::filesystem::exists(checkpoint.path()));
  metadata = checkpoint_metadata();
  Kokkos::deep_copy(execution, storage.flat_state(),
                    teuk::Complex(
                        std::numeric_limits<double>::quiet_NaN(), 0.0));
  rejected = false;
  try {
    static_cast<void>(teuk::save_plus2_checkpoint(
        execution, checkpoint.path(), storage, metadata));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(!std::filesystem::exists(checkpoint.path()));
  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(7.0, -3.0));
  auto expected = checkpoint_expectations();
  expected.theta_coordinates[0] =
      std::numeric_limits<double>::infinity();
  rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_checkpoint(
        execution, checkpoint.path(), storage, expected));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  const auto unchanged = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < unchanged.extent(0); ++i)
    CHECK(unchanged(i) == teuk::Complex(7.0, -3.0));
}

TEST_CASE("plus2 legacy checkpoints lacking physical provenance are rejected") {
  const teuk::ExecutionSpace execution;
  auto storage = teuk::Plus2CompanionStorage::enabled(
      3, 2, 2, "plus2_legacy_checkpoint_test");
  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex(1.0, -2.0));
  TemporaryCheckpoint checkpoint;
  static_cast<void>(teuk::save_plus2_checkpoint(
      execution, checkpoint.path(), storage, checkpoint_metadata()));
  convert_plus2_checkpoint_to_legacy_v2(checkpoint.path());

  const auto expected = checkpoint_expectations();
  Kokkos::deep_copy(execution, storage.flat_state(),
                    teuk::Complex(9.0, -4.0));
  bool rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_checkpoint(
        execution, checkpoint.path(), storage, expected));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  const auto unchanged = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < unchanged.extent(0); ++i) {
    CHECK(unchanged(i) == teuk::Complex(9.0, -4.0));
  }

  TemporaryCheckpoint legacy_v3;
  auto legacy_metadata = checkpoint_metadata();
  legacy_metadata.version =
      teuk::plus2_checkpoint_legacy_unbound_pde_version;
  legacy_metadata.radial_count = storage.radial_count();
  legacy_metadata.theta_count = storage.theta_count();
  std::vector<teuk::Complex> legacy_values(storage.value_count(),
                                            teuk::Complex(1.0, -2.0));
  legacy_metadata.state_checksum =
      teuk::plus2_checkpoint_detail::checksum(legacy_values);
  {
    std::ofstream output(legacy_v3.path(),
                         std::ios::binary | std::ios::trunc);
    teuk::plus2_checkpoint_detail::write_payload(
        output, legacy_metadata, legacy_values);
  }
  Kokkos::deep_copy(execution, storage.flat_state(),
                    teuk::Complex(6.0, -5.0));
  rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_checkpoint(
        execution, legacy_v3.path(), storage, expected));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  const auto unchanged_v3 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, storage.flat_state());
  for (std::size_t i = 0; i < unchanged_v3.extent(0); ++i) {
    CHECK(unchanged_v3(i) == teuk::Complex(6.0, -5.0));
  }
}

TEST_CASE("config-only plus2 checkpoint restore is rejected before mutation") {
  const teuk::ExecutionSpace execution;
  TemporaryCheckpoint checkpoint;
  auto storage = teuk::Plus2CompanionStorage::enabled(
      3, 1, 1, "plus2_time_bound_checkpoint");
  Kokkos::deep_copy(execution, storage.flat_state(), teuk::Complex{});
  auto metadata = checkpoint_metadata();
  metadata.initial_policy = teuk::Plus2InitialPolicy::Checkpoint;
  metadata.radial_coordinates = {0.0};
  metadata.theta_coordinates = {0.25};
  metadata.radial_count = 1;
  metadata.theta_count = 1;
  metadata.dissipation = 0.0;
  metadata.progress = {0.2, 2};
  metadata.source_activation = {true, 0.1, 2, 0.2};
  static_cast<void>(teuk::save_plus2_checkpoint(
      execution, checkpoint.path(), storage, metadata));

  auto config = concurrent_configuration();
  config.initial_policy = teuk::Plus2InitialPolicy::Checkpoint;
  config.checkpoint = checkpoint.path();
  teuk::Plus2ReplayOrchestrator orchestrator(config);
  Kokkos::deep_copy(execution, orchestrator.companion_state(),
                    teuk::Complex(6.0, -2.0));
  execution.fence("set config-only restore mutation sentinel");
  bool rejected = false;
  try {
    static_cast<void>(orchestrator.initialize_checkpoint(execution));
  } catch (const std::logic_error&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(!orchestrator.initialized());
  const auto unchanged = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, orchestrator.companion_state());
  for (std::size_t index = 0; index < unchanged.extent(0); ++index) {
    CHECK(unchanged(index) == teuk::Complex(6.0, -2.0));
  }
}

TEST_CASE("plus2 checkpoint rejects representation metadata corruption") {
  const teuk::ExecutionSpace execution;
  auto storage = teuk::Plus2CompanionStorage::enabled(
      3, 2, 2, "plus2_representation_checkpoint_test");
  const auto expectations = checkpoint_expectations();
  struct Corruption {
    std::string original;
    std::string replacement;
  };
  const std::vector<Corruption> corruptions{
      {teuk::plus2_native_byte_order(),
       std::string(std::string(teuk::plus2_native_byte_order()).size(), 'x')},
      {teuk::plus2_binary64_format, "IEEE-754-binary32"},
      {teuk::plus2_complex_component_order, "imag-then-real"},
      {teuk::plus2_state_storage_order,
       "LayoutRight(mode,field,radial,theta);field-order=(Z,Q,P)"}};

  for (const auto& corruption : corruptions) {
    Kokkos::deep_copy(execution, storage.flat_state(),
                      teuk::Complex(1.25, -0.5));
    TemporaryCheckpoint checkpoint;
    static_cast<void>(teuk::save_plus2_checkpoint(
        execution, checkpoint.path(), storage, checkpoint_metadata()));
    replace_checkpoint_bytes(checkpoint.path(), corruption.original,
                             corruption.replacement);
    Kokkos::deep_copy(execution, storage.flat_state(),
                      teuk::Complex(-8.0, 3.0));
    execution.fence("set representation-corruption mutation sentinel");
    bool rejected = false;
    try {
      static_cast<void>(teuk::load_plus2_checkpoint(
          execution, checkpoint.path(), storage, expectations));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
    const auto unchanged = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, storage.flat_state());
    for (std::size_t i = 0; i < unchanged.extent(0); ++i) {
      CHECK(unchanged(i) == teuk::Complex(-8.0, 3.0));
    }
  }
}

TEST_CASE("plus2 step rejects inconsistent accepted activation before RHS") {
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_invalid_activation_primary", 2);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  teuk::Plus2ReplayOrchestrator orchestrator(concurrent_configuration());
  orchestrator.initialize_zero(execution);
  Kokkos::deep_copy(execution, primary, teuk::Complex(4.0, -1.0));
  Kokkos::deep_copy(execution, orchestrator.companion_state(),
                    teuk::Complex(-2.0, 3.0));
  execution.fence("set invalid activation mutation sentinels");
  const auto primary_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, orchestrator.companion_state());
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
  rejected = false;
  try {
    orchestrator.advance_concurrent(
        execution, primary, 0.2, 0.1,
        teuk::SourceActivationState{
            false, -1.0, 0,
            std::nextafter(-1.0,
                           -std::numeric_limits<double>::infinity())},
        primary_rhs,
        companion_rhs, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(calls == 0);
  const auto primary_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, orchestrator.companion_state());
  for (std::size_t index = 0; index < primary_before.extent(0); ++index) {
    CHECK(primary_after(index) == primary_before(index));
  }
  for (std::size_t index = 0; index < companion_before.extent(0); ++index) {
    CHECK(companion_after(index) == companion_before(index));
  }

  teuk::Plus2ReplayOrchestrator exact_sentinel(concurrent_configuration());
  exact_sentinel.initialize_zero(execution);
  int sentinel_calls = 0;
  const auto zero_primary_rhs = [&](const auto& stage_execution, double,
                                    const auto&, const auto& output) {
    ++sentinel_calls;
    Kokkos::deep_copy(stage_execution, output, teuk::Complex{});
  };
  const auto zero_companion_rhs =
      [&](const auto& stage_execution, double, const auto&, const auto&,
          const auto& output, const auto&) {
        ++sentinel_calls;
        Kokkos::deep_copy(stage_execution, output, teuk::Complex{});
      };
  exact_sentinel.advance_concurrent(
      execution, primary, 0.0, 0.1,
      teuk::SourceActivationState{false, -1.0, 0, -1.0}, zero_primary_rhs,
      zero_companion_rhs, workspace);
  CHECK(sentinel_calls == 8);
}
