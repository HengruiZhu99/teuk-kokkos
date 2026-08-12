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
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/checkpoint_identity.hpp"
#include "teuk/plus2_companion_storage.hpp"
#include "teuk/plus2_runtime_types.hpp"
#include "teuk/plus2_source.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

class Plus2CompanionPipeline;

inline constexpr std::uint32_t plus2_checkpoint_format_version = 4;
inline constexpr std::uint32_t plus2_checkpoint_legacy_unbound_pde_version =
    3;
inline constexpr std::uint32_t plus2_checkpoint_legacy_representation_version =
    2;
inline constexpr std::uint32_t plus2_checkpoint_legacy_d42_version = 1;
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
inline constexpr const char* plus2_provenance_binding_schema =
    "plus2-pipeline-derived-v1";
inline constexpr const char* plus2_unbound_codec_schema =
    "plus2-unbound-codec-v1";

class Plus2PipelineCheckpointAuthority {
 private:
  explicit constexpr Plus2PipelineCheckpointAuthority(const int) {}
  friend class Plus2CompanionPipeline;
};

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
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  double mass = 0.0;
  double spin = 0.0;
  double compactification_length = 0.0;
  std::vector<double> radial_coordinates;
  std::vector<double> theta_coordinates;
  double time_step = 0.0;
  std::string reduction_mode;
  double reduction_damping = 0.0;
  double dissipation = 0.0;
  std::string provenance_binding_schema = plus2_unbound_codec_schema;
  Plus2SourceNormalization source_normalization =
      plus2_source_normalization;
  PrimaryCheckpointContentIdentity primary_checkpoint_identity;
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
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  double mass = 0.0;
  double spin = 0.0;
  double compactification_length = 0.0;
  std::vector<double> radial_coordinates;
  std::vector<double> theta_coordinates;
  double time_step = 0.0;
  std::string reduction_mode;
  double reduction_damping = 0.0;
  double dissipation = 0.0;
  std::string provenance_binding_schema = plus2_unbound_codec_schema;
  Plus2SourceNormalization source_normalization =
      plus2_source_normalization;
  PrimaryCheckpointContentIdentity primary_checkpoint_identity;
  Plus2CheckpointProgress progress;
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
      activation.last_eligibility_time < -1.0 ||
      activation.last_eligibility_time > progress_time) {
    throw std::invalid_argument(
        "invalid accepted source-activation state in plus2 checkpoint");
  }
}

inline void validate_physical_problem(
    const double mass, const double spin, const double length,
    const std::vector<double>& radial_coordinates,
    const std::vector<double>& theta_coordinates, const double time_step,
    const std::string& reduction_mode, const double reduction_damping,
    const double dissipation, const std::string& provenance_binding,
    const Plus2SourceNormalization normalization,
    const PrimaryCheckpointContentIdentity& primary_checkpoint_identity,
    const std::size_t radial_count, const std::size_t theta_count,
    const bool require_resolved_step = true,
    const bool require_primary_identity = true,
    const bool require_pipeline_binding = true) {
  const auto finite_coordinates = [](const std::vector<double>& coordinates) {
    return std::all_of(coordinates.begin(), coordinates.end(),
                       [](const double value) { return std::isfinite(value); });
  };
  const bool radial_strict =
      std::adjacent_find(radial_coordinates.begin(), radial_coordinates.end(),
                         std::greater_equal<double>()) ==
      radial_coordinates.end();
  const bool theta_increasing =
      std::adjacent_find(theta_coordinates.begin(), theta_coordinates.end(),
                         std::greater_equal<double>()) ==
      theta_coordinates.end();
  const bool theta_decreasing =
      std::adjacent_find(theta_coordinates.begin(), theta_coordinates.end(),
                         std::less_equal<double>()) ==
      theta_coordinates.end();
  if (!std::isfinite(mass) || !std::isfinite(spin) ||
      !std::isfinite(length) || mass <= 0.0 || std::abs(spin) > mass ||
      length <= 0.0 || radial_coordinates.size() != radial_count ||
      theta_coordinates.size() != theta_count || !finite_coordinates(radial_coordinates) ||
      !finite_coordinates(theta_coordinates) || !radial_strict ||
      (!theta_increasing && !theta_decreasing) ||
      !std::isfinite(time_step) || time_step < 0.0 ||
      (require_resolved_step && time_step == 0.0) ||
      (reduction_mode != "free_damped" &&
       reduction_mode != "stage_constrained") ||
      !std::isfinite(reduction_damping) || reduction_damping < 0.0 ||
      !std::isfinite(dissipation) || dissipation < 0.0 ||
      (require_pipeline_binding
           ? provenance_binding != plus2_provenance_binding_schema
           : provenance_binding != plus2_unbound_codec_schema) ||
      normalization != plus2_source_normalization) {
    throw std::invalid_argument(
        "incomplete or nonfinite plus2 physical-problem provenance");
  }
  (void)plus2_source_normalization_name_of(normalization);
  if (require_primary_identity) {
    validate_primary_checkpoint_content_identity(primary_checkpoint_identity);
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
  (void)radial_discretization_name(expected.radial_discretization);
  validate_physical_problem(
      expected.mass, expected.spin, expected.compactification_length,
      expected.radial_coordinates, expected.theta_coordinates,
      expected.time_step, expected.reduction_mode,
      expected.reduction_damping, expected.dissipation,
      expected.provenance_binding_schema, expected.source_normalization,
      expected.primary_checkpoint_identity, expected.radial_count,
      expected.theta_count, true, true,
      expected.provenance_binding_schema == plus2_provenance_binding_schema);
  if (!std::isfinite(expected.progress.time) || expected.progress.time < 0.0) {
    throw std::invalid_argument("invalid expected plus2 checkpoint progress");
  }
}

inline void validate_metadata(const Plus2CheckpointMetadata& metadata) {
  if (metadata.schema != plus2_checkpoint_schema ||
      (metadata.version != plus2_checkpoint_format_version &&
       metadata.version != plus2_checkpoint_legacy_unbound_pde_version &&
       metadata.version != plus2_checkpoint_legacy_representation_version &&
       metadata.version != plus2_checkpoint_legacy_d42_version) ||
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
  (void)radial_discretization_name(metadata.radial_discretization);
  if (metadata.version == plus2_checkpoint_format_version) {
    const bool pipeline_bound =
        metadata.provenance_binding_schema == plus2_provenance_binding_schema;
    validate_physical_problem(
        metadata.mass, metadata.spin, metadata.compactification_length,
        metadata.radial_coordinates, metadata.theta_coordinates,
        metadata.time_step, metadata.reduction_mode,
        metadata.reduction_damping, metadata.dissipation,
        metadata.provenance_binding_schema, metadata.source_normalization,
        metadata.primary_checkpoint_identity, metadata.radial_count,
        metadata.theta_count, true, true, pipeline_bound);
    const double expected_time =
        static_cast<double>(metadata.progress.step) * metadata.time_step;
    const double scale =
        std::max({1.0, std::abs(expected_time),
                  std::abs(metadata.progress.time)});
    if (!std::isfinite(expected_time) ||
        std::abs(expected_time - metadata.progress.time) >
            8.0 * std::numeric_limits<double>::epsilon() * scale) {
      throw std::invalid_argument(
          "plus2 checkpoint time is inconsistent with step and dt");
    }
  }
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

inline void require_finite_state(const std::vector<Complex>& values) {
  for (const Complex value : values) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      throw std::runtime_error("nonfinite plus2 checkpoint state");
    }
  }
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

inline void write_coordinates(std::ostream& output,
                              const std::vector<double>& coordinates) {
  write_scalar(output, static_cast<std::uint64_t>(coordinates.size()));
  for (const double coordinate : coordinates) write_scalar(output, coordinate);
}

inline std::vector<double> read_coordinates(std::istream& input,
                                            const char* label) {
  constexpr std::uint64_t maximum_coordinate_count = 1ULL << 24;
  const auto count = read_scalar<std::uint64_t>(input, label);
  if (count > maximum_coordinate_count) {
    throw std::runtime_error("unreasonable coordinate count in plus2 checkpoint");
  }
  std::vector<double> coordinates(static_cast<std::size_t>(count));
  for (double& coordinate : coordinates) {
    coordinate = read_scalar<double>(input, label);
  }
  return coordinates;
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
  if (metadata.version >= plus2_checkpoint_legacy_representation_version) {
    write_string(output,
                 radial_discretization_name(metadata.radial_discretization));
  }
  if (metadata.version >= plus2_checkpoint_legacy_unbound_pde_version) {
    write_scalar(output, metadata.mass);
    write_scalar(output, metadata.spin);
    write_scalar(output, metadata.compactification_length);
    write_coordinates(output, metadata.radial_coordinates);
    write_coordinates(output, metadata.theta_coordinates);
    write_scalar(output, metadata.time_step);
    write_string(output, metadata.reduction_mode);
    write_scalar(output, metadata.reduction_damping);
    write_scalar(output, metadata.dissipation);
    if (metadata.version == plus2_checkpoint_format_version) {
      write_string(output, metadata.provenance_binding_schema);
      write_scalar(output,
                   static_cast<std::uint32_t>(metadata.source_normalization));
      write_string(output,
                   plus2_source_normalization_name_of(
                       metadata.source_normalization));
      write_string(output, metadata.primary_checkpoint_identity.algorithm);
      write_string(output, metadata.primary_checkpoint_identity.state_schema);
      write_string(output,
                   sha256_hex(metadata.primary_checkpoint_identity.digest));
    } else {
      write_scalar(output, plus2_source_normalization_version);
      write_string(output, plus2_source_normalization_name);
      write_string(output, "legacy-unbound-primary-identity");
    }
  }
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
  metadata.radial_discretization =
      metadata.version == plus2_checkpoint_legacy_d42_version
          ? RadialDiscretization::D42
          : parse_radial_discretization(
                read_string(input, "radial discretization"));
  if (metadata.version >= plus2_checkpoint_legacy_unbound_pde_version) {
    metadata.mass = read_scalar<double>(input, "mass");
    metadata.spin = read_scalar<double>(input, "spin");
    metadata.compactification_length =
        read_scalar<double>(input, "compactification length");
    metadata.radial_coordinates = read_coordinates(input, "radial coordinates");
    metadata.theta_coordinates = read_coordinates(input, "theta coordinates");
    metadata.time_step = read_scalar<double>(input, "time step");
    metadata.reduction_mode = read_string(input, "reduction mode");
    metadata.reduction_damping =
        read_scalar<double>(input, "reduction damping");
    metadata.dissipation = read_scalar<double>(input, "dissipation");
    if (metadata.version == plus2_checkpoint_format_version) {
      metadata.provenance_binding_schema =
          read_string(input, "provenance binding schema");
      const auto source_version =
          read_scalar<std::uint32_t>(input, "source normalization version");
      const auto source_name = read_string(input, "source normalization name");
      metadata.source_normalization =
          parse_plus2_source_normalization(source_version, source_name);
      metadata.primary_checkpoint_identity.algorithm =
          read_string(input, "primary identity algorithm");
      metadata.primary_checkpoint_identity.state_schema =
          read_string(input, "primary identity state schema");
      metadata.primary_checkpoint_identity.digest = parse_sha256_hex(
          read_string(input, "primary identity digest"));
    } else {
      const auto source_version =
          read_scalar<std::uint32_t>(input, "source normalization version");
      const auto source_name = read_string(input, "source normalization name");
      metadata.source_normalization =
          parse_plus2_source_normalization(source_version, source_name);
      static_cast<void>(read_string(input, "legacy primary identity"));
      metadata.provenance_binding_schema.clear();
      metadata.primary_checkpoint_identity = {};
    }
  }
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
  require_finite_state(values);
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
      metadata.theta_count != expected.theta_count ||
      metadata.radial_discretization != expected.radial_discretization ||
      metadata.version != plus2_checkpoint_format_version ||
      metadata.mass != expected.mass || metadata.spin != expected.spin ||
      metadata.compactification_length != expected.compactification_length ||
      metadata.radial_coordinates != expected.radial_coordinates ||
      metadata.theta_coordinates != expected.theta_coordinates ||
      metadata.time_step != expected.time_step ||
      metadata.reduction_mode != expected.reduction_mode ||
      metadata.reduction_damping != expected.reduction_damping ||
      metadata.dissipation != expected.dissipation ||
      metadata.provenance_binding_schema !=
          expected.provenance_binding_schema ||
      metadata.source_normalization != expected.source_normalization ||
      metadata.primary_checkpoint_identity !=
          expected.primary_checkpoint_identity ||
      metadata.progress.time != expected.progress.time ||
      metadata.progress.step != expected.progress.step) {
    throw std::runtime_error(
        "plus2 checkpoint does not match scaling, registry, methods, provenance, or shape");
  }
}

}  // namespace plus2_checkpoint_detail

namespace plus2_checkpoint_detail {

inline Plus2CheckpointMetadata save_checkpoint_impl(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    const Plus2CompanionStorage& storage, Plus2CheckpointMetadata metadata,
    const std::string_view required_binding) {
  if (!storage.is_enabled()) {
    throw std::invalid_argument("cannot checkpoint disabled plus2 storage");
  }
  if (metadata.version != plus2_checkpoint_format_version) {
    throw std::invalid_argument(
        "new plus2 checkpoints must use the current format version");
  }
  if (metadata.provenance_binding_schema != required_binding) {
    throw std::invalid_argument(
        "plus2 checkpoint writer lacks the required binding authority");
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
  require_finite_state(values);
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
inline Plus2CheckpointMetadata load_checkpoint_impl(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    Plus2CompanionStorage& storage,
    const Plus2CheckpointExpectations& expected,
    const std::string_view required_binding) {
  if (!storage.is_enabled()) {
    throw std::invalid_argument("cannot restore disabled plus2 storage");
  }
  validate_expectations(expected);
  if (expected.provenance_binding_schema != required_binding) {
    throw std::invalid_argument(
        "plus2 checkpoint loader lacks the required binding authority");
  }
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

}  // namespace plus2_checkpoint_detail

inline Plus2CheckpointMetadata save_plus2_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    const Plus2CompanionStorage& storage, Plus2CheckpointMetadata metadata) {
  return plus2_checkpoint_detail::save_checkpoint_impl(
      execution, path, storage, std::move(metadata), plus2_unbound_codec_schema);
}

inline Plus2CheckpointMetadata save_plus2_pipeline_checkpoint(
    const Plus2PipelineCheckpointAuthority,
    const ExecutionSpace& execution, const std::filesystem::path& path,
    const Plus2CompanionStorage& storage, Plus2CheckpointMetadata metadata) {
  return plus2_checkpoint_detail::save_checkpoint_impl(
      execution, path, storage, std::move(metadata),
      plus2_provenance_binding_schema);
}

inline Plus2CheckpointMetadata load_plus2_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& path,
    Plus2CompanionStorage& storage,
    const Plus2CheckpointExpectations& expected) {
  return plus2_checkpoint_detail::load_checkpoint_impl(
      execution, path, storage, expected, plus2_unbound_codec_schema);
}

inline Plus2CheckpointMetadata load_plus2_pipeline_checkpoint(
    const Plus2PipelineCheckpointAuthority,
    const ExecutionSpace& execution, const std::filesystem::path& path,
    Plus2CompanionStorage& storage,
    const Plus2CheckpointExpectations& expected) {
  return plus2_checkpoint_detail::load_checkpoint_impl(
      execution, path, storage, expected, plus2_provenance_binding_schema);
}

}  // namespace teuk
