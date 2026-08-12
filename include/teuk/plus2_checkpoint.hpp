#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/plus2_companion_storage.hpp"
#include "teuk/plus2_runtime_types.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr std::uint32_t plus2_checkpoint_format_version = 1;
inline constexpr const char* plus2_checkpoint_schema =
    "teuk.plus2-companion-checkpoint";
inline constexpr const char* plus2_fixed_tetrad_raw_scaling =
    "Psi0_raw_fixed_tetrad=(R^5/(L^2-i*a*R*cos(theta))^4)*Z_plus;"
    "Z0_source=Psi0_raw_fixed_tetrad/R^5="
    "Z_plus/(L^2-i*a*R*cos(theta))^4";
inline constexpr const char* plus2_signed_mode_registry =
    "signed-m-sharp-registry-v1";
inline constexpr const char* plus2_binary64_format = "IEEE-754-binary64";
inline constexpr const char* plus2_complex_component_order =
    "real-then-imag";
inline constexpr const char* plus2_state_storage_order =
    "LayoutRight(mode,field,radial,theta);field-order=(P,Q,Z)";

#ifndef TEUK_GIT_COMMIT
#define TEUK_GIT_COMMIT "unknown"
#endif

inline const char* plus2_build_git_commit() { return TEUK_GIT_COMMIT; }

inline const char* plus2_native_byte_order() {
  if constexpr (std::endian::native == std::endian::little) return "little";
  if constexpr (std::endian::native == std::endian::big) return "big";
  throw std::runtime_error(
      "mixed-endian plus2 checkpoint storage is unsupported");
}

struct Plus2CheckpointProgress {
  double time = 0.0;
  std::uint64_t step = 0;
};

struct Plus2CheckpointMetadata {
  std::string schema = plus2_checkpoint_schema;
  std::uint32_t version = plus2_checkpoint_format_version;
  std::string byte_order = plus2_native_byte_order();
  std::string floating_point_format = plus2_binary64_format;
  std::string complex_component_order = plus2_complex_component_order;
  std::string state_storage_order = plus2_state_storage_order;
  std::string scaling = plus2_fixed_tetrad_raw_scaling;
  std::string registry_schema = plus2_signed_mode_registry;
  std::vector<int> parent_modes;
  std::vector<int> target_modes;
  int ell_max_first = 0;
  int ell_max_second = 0;
  Plus2LinearMethod linear_method = Plus2LinearMethod::MetricCurvature;
  Plus2SecondMethod second_method = Plus2SecondMethod::SourcedCompanion;
  Plus2InitialPolicy initial_policy = Plus2InitialPolicy::Zero;
  std::string git_commit;
  int runtime_config_schema_version = 0;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
  Plus2CheckpointProgress progress;
  SourceActivationState source_activation;
  std::uint64_t state_checksum = 0;
};

struct Plus2CheckpointExpectations {
  std::string scaling = plus2_fixed_tetrad_raw_scaling;
  std::string registry_schema = plus2_signed_mode_registry;
  std::vector<int> parent_modes;
  std::vector<int> target_modes;
  int ell_max_first = 0;
  int ell_max_second = 0;
  Plus2LinearMethod linear_method = Plus2LinearMethod::MetricCurvature;
  Plus2SecondMethod second_method = Plus2SecondMethod::SourcedCompanion;
  Plus2InitialPolicy initial_policy = Plus2InitialPolicy::Zero;
  std::string git_commit;
  int runtime_config_schema_version = 0;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
};

namespace plus2_checkpoint_detail {

inline constexpr std::array<char, 8> magic{'T', 'E', 'U', 'K', 'P', '2', 'C',
                                           'P'};

inline void require_sorted_sharp_registry(const std::vector<int>& modes,
                                          const char* label) {
  if (modes.empty() || !std::is_sorted(modes.begin(), modes.end()) ||
      std::adjacent_find(modes.begin(), modes.end()) != modes.end()) {
    throw std::invalid_argument(std::string(label) +
                                " must be nonempty, sorted, and unique");
  }
  for (const int mode : modes) {
    if (!std::binary_search(modes.begin(), modes.end(), -mode)) {
      throw std::invalid_argument(std::string(label) +
                                  " must be closed under sharp");
    }
  }
}

inline void require_registry_band(const std::vector<int>& modes,
                                  const int ell_max, const char* label) {
  if (ell_max < 2) {
    throw std::invalid_argument(std::string(label) +
                                " ell_max must be at least 2");
  }
  for (const int mode : modes) {
    if (std::abs(mode) > ell_max) {
      throw std::invalid_argument(std::string(label) +
                                  " contains a mode above ell_max");
    }
  }
}

inline void require_git_commit(const std::string& commit) {
  const bool hexadecimal =
      !commit.empty() &&
      std::all_of(commit.begin(), commit.end(), [](const unsigned char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f') ||
               (value >= 'A' && value <= 'F');
      });
  if (!hexadecimal || commit.size() < 7) {
    throw std::invalid_argument(
        "plus2 checkpoint requires an explicit Git commit identifier");
  }
}

inline void validate_scientific_configuration(
    const std::vector<int>& parent_modes,
    const std::vector<int>& target_modes, const int ell_max_first,
    const int ell_max_second, const Plus2LinearMethod linear_method,
    const Plus2SecondMethod second_method,
    const Plus2InitialPolicy initial_policy, const std::string& git_commit,
    const int runtime_schema_version) {
  require_registry_band(parent_modes, ell_max_first, "parent registry");
  require_registry_band(target_modes, ell_max_second, "target registry");
  (void)plus2_linear_method_name(linear_method);
  (void)plus2_second_method_name(second_method);
  (void)plus2_initial_policy_name(initial_policy);
  require_git_commit(git_commit);
  if (runtime_schema_version <= 0) {
    throw std::invalid_argument(
        "plus2 checkpoint runtime config schema version must be positive");
  }
}

inline void validate_activation(const SourceActivationState& activation,
                                const double progress_time) {
  if (!std::isfinite(progress_time) || progress_time < 0.0 ||
      !std::isfinite(activation.activation_time) ||
      !std::isfinite(activation.last_eligibility_time) ||
      activation.consecutive_passes < 0 ||
      (activation.active &&
       (activation.activation_time < 0.0 ||
        activation.activation_time > progress_time)) ||
      (!activation.active && activation.activation_time != -1.0) ||
      activation.last_eligibility_time > progress_time) {
    throw std::invalid_argument(
        "invalid accepted source-activation state in plus2 checkpoint");
  }
}

inline void validate_expectations(const Plus2CheckpointExpectations& expected) {
  if (expected.scaling.empty() || expected.registry_schema.empty() ||
      expected.radial_count == 0 || expected.theta_count == 0) {
    throw std::invalid_argument("incomplete plus2 checkpoint expectations");
  }
  require_sorted_sharp_registry(expected.parent_modes, "parent registry");
  require_sorted_sharp_registry(expected.target_modes, "target registry");
  validate_scientific_configuration(
      expected.parent_modes, expected.target_modes, expected.ell_max_first,
      expected.ell_max_second, expected.linear_method, expected.second_method,
      expected.initial_policy, expected.git_commit,
      expected.runtime_config_schema_version);
}

inline void validate_metadata(const Plus2CheckpointMetadata& metadata) {
  if (metadata.schema != plus2_checkpoint_schema ||
      metadata.version != plus2_checkpoint_format_version ||
      metadata.byte_order != plus2_native_byte_order() ||
      metadata.floating_point_format != plus2_binary64_format ||
      metadata.complex_component_order != plus2_complex_component_order ||
      metadata.state_storage_order != plus2_state_storage_order ||
      metadata.scaling.empty() || metadata.registry_schema.empty() ||
      metadata.radial_count == 0 || metadata.theta_count == 0) {
    throw std::invalid_argument(
        "invalid plus2 checkpoint schema, representation, or shape");
  }
  if (sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559 ||
      std::numeric_limits<double>::digits != 53 ||
      std::numeric_limits<double>::max_exponent != 1024) {
    throw std::runtime_error(
        "host does not provide the required IEEE-754 binary64 format");
  }
  require_sorted_sharp_registry(metadata.parent_modes, "parent registry");
  require_sorted_sharp_registry(metadata.target_modes, "target registry");
  validate_scientific_configuration(
      metadata.parent_modes, metadata.target_modes, metadata.ell_max_first,
      metadata.ell_max_second, metadata.linear_method, metadata.second_method,
      metadata.initial_policy, metadata.git_commit,
      metadata.runtime_config_schema_version);
  validate_activation(metadata.source_activation, metadata.progress.time);
}

inline std::size_t checked_value_count(const Plus2CheckpointMetadata& metadata) {
  constexpr std::size_t fields =
      static_cast<std::size_t>(TeukolskyField::Count);
  const std::array<std::size_t, 4> factors{
      metadata.target_modes.size(), fields, metadata.radial_count,
      metadata.theta_count};
  std::size_t result = 1;
  for (const std::size_t factor : factors) {
    if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::overflow_error("plus2 checkpoint state size overflow");
    }
    result *= factor;
  }
  return result;
}

inline std::uint64_t checksum(const std::vector<Complex>& values) {
  static_assert(sizeof(double) == 8,
                "plus2 checkpoints require binary64 doubles");
  constexpr std::uint64_t offset = 14695981039346656037ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  std::uint64_t result = offset;
  for (const Complex value : values) {
    for (const double component : {value.real(), value.imag()}) {
      const auto bytes =
          std::bit_cast<std::array<unsigned char, sizeof(double)>>(component);
      for (const unsigned char byte : bytes) {
        result ^= static_cast<std::uint64_t>(byte);
        result *= prime;
      }
    }
  }
  return result;
}

template <class Value>
void write_scalar(std::ostream& output, const Value value) {
  static_assert(std::is_trivially_copyable_v<Value>);
  output.write(reinterpret_cast<const char*>(&value), sizeof(Value));
}

template <class Value>
Value read_scalar(std::istream& input, const char* label) {
  static_assert(std::is_trivially_copyable_v<Value>);
  Value value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(Value));
  if (!input) {
    throw std::runtime_error(std::string("truncated plus2 checkpoint at ") +
                             label);
  }
  return value;
}

inline void write_string(std::ostream& output, const std::string& value) {
  write_scalar(output, static_cast<std::uint64_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline std::string read_string(std::istream& input, const char* label) {
  constexpr std::uint64_t maximum_string_size = 4096;
  const auto size = read_scalar<std::uint64_t>(input, label);
  if (size > maximum_string_size) {
    throw std::runtime_error("unreasonable string length in plus2 checkpoint");
  }
  std::string value(static_cast<std::size_t>(size), '\0');
  input.read(value.data(), static_cast<std::streamsize>(value.size()));
  if (!input) {
    throw std::runtime_error(std::string("truncated plus2 checkpoint at ") +
                             label);
  }
  return value;
}

inline void write_modes(std::ostream& output, const std::vector<int>& modes) {
  write_scalar(output, static_cast<std::uint64_t>(modes.size()));
  for (const int mode : modes) write_scalar(output, static_cast<std::int32_t>(mode));
}

inline std::vector<int> read_modes(std::istream& input, const char* label) {
  constexpr std::uint64_t maximum_mode_count = 1ULL << 20;
  const auto count = read_scalar<std::uint64_t>(input, label);
  if (count > maximum_mode_count) {
    throw std::runtime_error("unreasonable registry size in plus2 checkpoint");
  }
  std::vector<int> modes(static_cast<std::size_t>(count));
  for (int& mode : modes) {
    mode = static_cast<int>(read_scalar<std::int32_t>(input, label));
  }
  return modes;
}

inline void write_payload(std::ostream& output,
                          const Plus2CheckpointMetadata& metadata,
                          const std::vector<Complex>& values) {
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  write_string(output, metadata.schema);
  write_scalar(output, metadata.version);
  write_string(output, metadata.byte_order);
  write_string(output, metadata.floating_point_format);
  write_string(output, metadata.complex_component_order);
  write_string(output, metadata.state_storage_order);
  write_string(output, metadata.scaling);
  write_string(output, metadata.registry_schema);
  write_modes(output, metadata.parent_modes);
  write_modes(output, metadata.target_modes);
  write_scalar(output, static_cast<std::int32_t>(metadata.ell_max_first));
  write_scalar(output, static_cast<std::int32_t>(metadata.ell_max_second));
  write_string(output, plus2_linear_method_name(metadata.linear_method));
  write_string(output, plus2_second_method_name(metadata.second_method));
  write_string(output, plus2_initial_policy_name(metadata.initial_policy));
  write_string(output, metadata.git_commit);
  write_scalar(output, static_cast<std::int32_t>(
                           metadata.runtime_config_schema_version));
  write_scalar(output, static_cast<std::uint64_t>(metadata.radial_count));
  write_scalar(output, static_cast<std::uint64_t>(metadata.theta_count));
  write_scalar(output, metadata.progress.time);
  write_scalar(output, metadata.progress.step);
  write_scalar(output, static_cast<std::uint8_t>(metadata.source_activation.active));
  write_scalar(output, metadata.source_activation.activation_time);
  write_scalar(output, static_cast<std::int32_t>(
                           metadata.source_activation.consecutive_passes));
  write_scalar(output, metadata.source_activation.last_eligibility_time);
  write_scalar(output, metadata.state_checksum);
  write_scalar(output, static_cast<std::uint64_t>(values.size()));
  for (const Complex value : values) {
    write_scalar(output, value.real());
    write_scalar(output, value.imag());
  }
}

inline std::pair<Plus2CheckpointMetadata, std::vector<Complex>> read_payload(
    std::istream& input) {
  std::array<char, magic.size()> input_magic{};
  input.read(input_magic.data(), static_cast<std::streamsize>(input_magic.size()));
  if (!input || input_magic != magic) {
    throw std::runtime_error("invalid plus2 checkpoint magic");
  }
  Plus2CheckpointMetadata metadata;
  metadata.schema = read_string(input, "schema");
  metadata.version = read_scalar<std::uint32_t>(input, "version");
  metadata.byte_order = read_string(input, "byte order");
  metadata.floating_point_format = read_string(input, "floating-point format");
  metadata.complex_component_order =
      read_string(input, "complex component order");
  metadata.state_storage_order = read_string(input, "state storage order");
  metadata.scaling = read_string(input, "scaling");
  metadata.registry_schema = read_string(input, "registry schema");
  metadata.parent_modes = read_modes(input, "parent modes");
  metadata.target_modes = read_modes(input, "target modes");
  metadata.ell_max_first =
      static_cast<int>(read_scalar<std::int32_t>(input, "ell max first"));
  metadata.ell_max_second =
      static_cast<int>(read_scalar<std::int32_t>(input, "ell max second"));
  metadata.linear_method =
      parse_plus2_linear_method(read_string(input, "linear method"));
  metadata.second_method =
      parse_plus2_second_method(read_string(input, "second method"));
  metadata.initial_policy =
      parse_plus2_initial_policy(read_string(input, "initial policy"));
  metadata.git_commit = read_string(input, "git commit");
  metadata.runtime_config_schema_version = static_cast<int>(
      read_scalar<std::int32_t>(input, "runtime config schema version"));
  metadata.radial_count = static_cast<std::size_t>(
      read_scalar<std::uint64_t>(input, "radial count"));
  metadata.theta_count = static_cast<std::size_t>(
      read_scalar<std::uint64_t>(input, "theta count"));
  metadata.progress.time = read_scalar<double>(input, "time");
  metadata.progress.step = read_scalar<std::uint64_t>(input, "step");
  const auto active = read_scalar<std::uint8_t>(input, "source active");
  if (active > 1) throw std::runtime_error("invalid source-active flag");
  metadata.source_activation.active = active != 0;
  metadata.source_activation.activation_time =
      read_scalar<double>(input, "source activation time");
  metadata.source_activation.consecutive_passes = static_cast<int>(
      read_scalar<std::int32_t>(input, "source consecutive passes"));
  metadata.source_activation.last_eligibility_time =
      read_scalar<double>(input, "source eligibility time");
  metadata.state_checksum = read_scalar<std::uint64_t>(input, "checksum");
  const auto stored_count = read_scalar<std::uint64_t>(input, "state count");
  validate_metadata(metadata);
  const std::size_t expected_count = checked_value_count(metadata);
  if (stored_count != expected_count) {
    throw std::runtime_error("plus2 checkpoint state count mismatch");
  }
  std::vector<Complex> values(expected_count);
  for (Complex& value : values) {
    const double real = read_scalar<double>(input, "state real component");
    const double imag = read_scalar<double>(input, "state imaginary component");
    value = Complex(real, imag);
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("trailing data in plus2 checkpoint");
  }
  return {std::move(metadata), std::move(values)};
}

inline void require_metadata_match(
    const Plus2CheckpointMetadata& metadata,
    const Plus2CheckpointExpectations& expected) {
  if (metadata.scaling != expected.scaling ||
      metadata.registry_schema != expected.registry_schema ||
      metadata.parent_modes != expected.parent_modes ||
      metadata.target_modes != expected.target_modes ||
      metadata.ell_max_first != expected.ell_max_first ||
      metadata.ell_max_second != expected.ell_max_second ||
      metadata.linear_method != expected.linear_method ||
      metadata.second_method != expected.second_method ||
      metadata.initial_policy != expected.initial_policy ||
      metadata.git_commit != expected.git_commit ||
      metadata.runtime_config_schema_version !=
          expected.runtime_config_schema_version ||
      metadata.radial_count != expected.radial_count ||
      metadata.theta_count != expected.theta_count) {
    throw std::runtime_error(
        "plus2 checkpoint does not match scaling, registry, methods, provenance, or shape");
  }
}

}  // namespace plus2_checkpoint_detail

inline Plus2CheckpointMetadata save_plus2_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    const Plus2CompanionStorage& storage, Plus2CheckpointMetadata metadata) {
  using namespace plus2_checkpoint_detail;
  if (!storage.is_enabled()) {
    throw std::invalid_argument("cannot checkpoint disabled plus2 storage");
  }
  metadata.radial_count = storage.radial_count();
  metadata.theta_count = storage.theta_count();
  validate_metadata(metadata);
  if (metadata.target_modes.size() != storage.mode_count() ||
      checked_value_count(metadata) != storage.value_count()) {
    throw std::invalid_argument("plus2 checkpoint metadata/storage mismatch");
  }
  Kokkos::View<Complex*, Kokkos::HostSpace> host("plus2_checkpoint_write",
                                                  storage.value_count());
  Kokkos::deep_copy(execution, host, storage.flat_state());
  execution.fence("copy plus2 checkpoint state to host");
  std::vector<Complex> values(host.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  metadata.state_checksum = checksum(values);

  if (path.empty() || std::filesystem::exists(path)) {
    throw std::runtime_error("plus2 checkpoint target must be new and nonempty");
  }
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path temporary =
      path.string() + ".tmp." + std::to_string(nonce);
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create plus2 checkpoint");
    write_payload(output, metadata, values);
    output.flush();
    if (!output) throw std::runtime_error("failed writing plus2 checkpoint");
    output.close();
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  return metadata;
}

// Validation is deliberately completed against host-owned temporary data
// before the destination View is mutated.
inline Plus2CheckpointMetadata load_plus2_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    Plus2CompanionStorage& storage,
    const Plus2CheckpointExpectations& expected) {
  using namespace plus2_checkpoint_detail;
  if (!storage.is_enabled()) {
    throw std::invalid_argument("cannot restore disabled plus2 storage");
  }
  validate_expectations(expected);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open plus2 checkpoint");
  auto [metadata, values] = read_payload(input);
  require_metadata_match(metadata, expected);
  if (storage.mode_count() != metadata.target_modes.size() ||
      storage.radial_count() != metadata.radial_count ||
      storage.theta_count() != metadata.theta_count ||
      storage.value_count() != values.size()) {
    throw std::runtime_error("plus2 checkpoint storage extent mismatch");
  }
  if (checksum(values) != metadata.state_checksum) {
    throw std::runtime_error("plus2 checkpoint state checksum mismatch");
  }

  Kokkos::View<Complex*, Kokkos::HostSpace> host("plus2_checkpoint_read",
                                                  values.size());
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(execution, storage.flat_state(), host);
  execution.fence("restore plus2 checkpoint state");
  return metadata;
}

}  // namespace teuk
