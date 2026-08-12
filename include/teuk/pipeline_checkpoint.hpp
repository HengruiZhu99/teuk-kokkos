#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "teuk/background.hpp"
#include "teuk/checkpoint_identity.hpp"
#include "teuk/io.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_bands.hpp"
#include "teuk/pipeline_io.hpp"
#include "teuk/spatial_pipeline.hpp"
#include "teuk/teukolsky.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr std::uint64_t pipeline_checkpoint_format_version = 5;
inline constexpr std::uint64_t pipeline_checkpoint_legacy_d42_version = 4;
inline constexpr const char* pipeline_checkpoint_metadata_file =
    "metadata.txt";
inline constexpr const char* pipeline_checkpoint_state_file = "state.bin";

// Parameters that are not recoverable from SpatialPipeline's public storage
// interface are explicit caller-supplied checkpoint provenance. Loading
// requires the caller's expected parameters to match them exactly.
struct PipelineCheckpointConfiguration {
  KerrParameters background;
  int ell_max_first = 0;
  int ell_max_second = 0;
  int theta_nodes = 0;
  double reduction_damping = 0.0;
  double dissipation = 0.0;
  ReductionEvolution reduction = ReductionEvolution::FreeDamped;
  double time_step = 0.0;
  SecondOrderSourcePolicy source_policy;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
};

struct PipelineCheckpointProgress {
  double time = 0.0;
  std::uint64_t step = 0;
};

struct PipelineCheckpointMetadata {
  std::uint64_t version = pipeline_checkpoint_format_version;
  SnapshotShape shape;
  std::vector<int> modes;
  std::vector<int> parents;
  std::vector<int> targets;
  std::vector<double> radial_coordinates;
  std::vector<double> theta_coordinates;
  PipelineCheckpointConfiguration configuration;
  PipelineCheckpointProgress progress;
  SourceActivationState source_activation;
  std::uint64_t state_checksum = 0;
};

namespace pipeline_checkpoint_detail {

inline bool same_shape(const SnapshotShape& left, const SnapshotShape& right) {
  return left.modes == right.modes && left.fields == right.fields &&
         left.radial == right.radial && left.theta == right.theta;
}

inline const char* reduction_name(const ReductionEvolution reduction) {
  switch (reduction) {
    case ReductionEvolution::FreeDamped: return "free_damped";
    case ReductionEvolution::StageConstrained: return "stage_constrained";
  }
  throw std::invalid_argument("unsupported reduction evolution");
}

inline ReductionEvolution parse_reduction(const std::string& text) {
  if (text == "free_damped") return ReductionEvolution::FreeDamped;
  if (text == "stage_constrained") {
    return ReductionEvolution::StageConstrained;
  }
  throw std::runtime_error("unsupported checkpoint reduction evolution");
}

inline const char* source_mode_name(const SecondOrderSourceMode mode) {
  switch (mode) {
    case SecondOrderSourceMode::Disabled: return "disabled";
    case SecondOrderSourceMode::ConstraintAware: return "constraint_aware";
    case SecondOrderSourceMode::Unrestricted: return "unrestricted";
  }
  throw std::invalid_argument("unsupported second-order source mode");
}

inline SecondOrderSourceMode parse_source_mode(const std::string& text) {
  if (text == "disabled") return SecondOrderSourceMode::Disabled;
  if (text == "constraint_aware") {
    return SecondOrderSourceMode::ConstraintAware;
  }
  if (text == "unrestricted") return SecondOrderSourceMode::Unrestricted;
  throw std::runtime_error("unsupported checkpoint second-order source mode");
}

inline const char* native_byte_order() {
  if constexpr (std::endian::native == std::endian::little) return "little";
  if constexpr (std::endian::native == std::endian::big) return "big";
  throw std::runtime_error("mixed-endian checkpoint storage is unsupported");
}

inline void validate_finite(const double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

inline void validate_configuration(
    const PipelineCheckpointConfiguration& configuration) {
  validate_finite(configuration.background.mass, "checkpoint mass");
  validate_finite(configuration.background.spin, "checkpoint spin");
  validate_finite(configuration.background.compactification_length,
                  "checkpoint compactification length");
  validate_finite(configuration.reduction_damping,
                  "checkpoint reduction damping");
  validate_finite(configuration.dissipation, "checkpoint dissipation");
  validate_finite(configuration.time_step, "checkpoint time step");
  validate_finite(configuration.source_policy.source_start_time,
                  "checkpoint source start time");
  validate_finite(
      configuration.source_policy.normalized_constraint_tolerance,
      "checkpoint source constraint tolerance");
  if (!(configuration.background.mass > 0.0) ||
      std::abs(configuration.background.spin) >
          configuration.background.mass ||
      !(configuration.background.compactification_length > 0.0) ||
      configuration.ell_max_first < 3 ||
      configuration.ell_max_second < 3 ||
      configuration.theta_nodes <
          std::max({configuration.ell_max_first + 1,
                    configuration.ell_max_second + 1,
                    (2 * configuration.ell_max_first +
                     configuration.ell_max_second + 2) /
                        2}) ||
      configuration.reduction_damping < 0.0 ||
      configuration.dissipation < 0.0 || !(configuration.time_step > 0.0) ||
      configuration.source_policy.source_start_time < 0.0 ||
      configuration.source_policy.normalized_constraint_tolerance < 0.0 ||
      configuration.source_policy.required_consecutive_passes < 1) {
    throw std::invalid_argument("invalid pipeline checkpoint configuration");
  }
  (void)reduction_name(configuration.reduction);
  (void)source_mode_name(configuration.source_policy.mode);
  (void)radial_discretization_name(configuration.radial_discretization);
}

inline bool same_configuration(
    const PipelineCheckpointConfiguration& left,
    const PipelineCheckpointConfiguration& right) {
  return left.background.mass == right.background.mass &&
         left.background.spin == right.background.spin &&
         left.background.compactification_length ==
             right.background.compactification_length &&
         left.ell_max_first == right.ell_max_first &&
         left.ell_max_second == right.ell_max_second &&
         left.theta_nodes == right.theta_nodes &&
         left.reduction_damping == right.reduction_damping &&
         left.dissipation == right.dissipation &&
         left.reduction == right.reduction &&
         left.radial_discretization == right.radial_discretization &&
         left.time_step == right.time_step &&
         left.source_policy.mode == right.source_policy.mode &&
         left.source_policy.source_start_time ==
             right.source_policy.source_start_time &&
         left.source_policy.normalized_constraint_tolerance ==
             right.source_policy.normalized_constraint_tolerance &&
         left.source_policy.required_consecutive_passes ==
             right.source_policy.required_consecutive_passes;
}

inline bool same_source_policy(const SecondOrderSourcePolicy& left,
                               const SecondOrderSourcePolicy& right) {
  return left.mode == right.mode &&
         left.source_start_time == right.source_start_time &&
         left.normalized_constraint_tolerance ==
             right.normalized_constraint_tolerance &&
         left.required_consecutive_passes ==
             right.required_consecutive_passes;
}

inline void validate_source_activation_state(
    const SourceActivationState& state, const double progress_time) {
  if (state.consecutive_passes < 0 ||
      !std::isfinite(state.activation_time) ||
      !std::isfinite(state.last_eligibility_time) ||
      (state.active && (state.activation_time < 0.0 ||
                        state.activation_time > progress_time)) ||
      (!state.active && state.activation_time != -1.0) ||
      state.last_eligibility_time < -1.0 ||
      state.last_eligibility_time > progress_time) {
    throw std::runtime_error("invalid checkpoint source activation state");
  }
}

template <class Value>
Value parse_number(const std::map<std::string, std::string>& entries,
                   const char* key) {
  const auto found = entries.find(key);
  if (found == entries.end()) {
    throw std::runtime_error(std::string("missing checkpoint key: ") + key);
  }
  Value value{};
  const std::string& text = found->second;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size()) {
    throw std::runtime_error(std::string("invalid checkpoint value for ") +
                             key);
  }
  return value;
}

inline const std::string& require_text(
    const std::map<std::string, std::string>& entries, const char* key) {
  const auto found = entries.find(key);
  if (found == entries.end()) {
    throw std::runtime_error(std::string("missing checkpoint key: ") + key);
  }
  return found->second;
}

template <class Value>
std::vector<Value> parse_list(
    const std::map<std::string, std::string>& entries, const char* key) {
  const std::string& text = require_text(entries, key);
  std::vector<Value> values;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t delimiter = text.find(',', begin);
    const std::size_t end =
        delimiter == std::string::npos ? text.size() : delimiter;
    const std::string_view token(text.data() + begin, end - begin);
    Value value{};
    const auto result =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || result.ec != std::errc{} ||
        result.ptr != token.data() + token.size()) {
      throw std::runtime_error(std::string("invalid checkpoint list: ") + key);
    }
    values.push_back(value);
    if (delimiter == std::string::npos) break;
    begin = delimiter + 1;
  }
  return values;
}

template <class Value>
void write_list(std::ostream& output, const std::vector<Value>& values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ',';
    output << values[i];
  }
  output << '\n';
}

inline std::uint64_t checksum(const std::vector<Complex>& values) {
  static_assert(sizeof(double) == 8,
                "pipeline checkpoints require binary64 doubles");
  constexpr std::uint64_t offset = 14695981039346656037ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  std::uint64_t result = offset;
  const auto add_double = [&](const double value) {
    const auto bytes =
        std::bit_cast<std::array<unsigned char, sizeof(double)>>(value);
    for (const unsigned char byte : bytes) {
      result ^= static_cast<std::uint64_t>(byte);
      result *= prime;
    }
  };
  for (const Complex value : values) {
    add_double(value.real());
    add_double(value.imag());
  }
  return result;
}

template <class View>
std::vector<typename View::non_const_value_type> copy_rank_one_to_host(
    const ExecutionSpace& execution, const View& view,
    const std::string& label) {
  static_assert(View::rank == 1, "checkpoint coordinate view must be rank one");
  using Value = typename View::non_const_value_type;
  Kokkos::View<Value*, Kokkos::HostSpace> host(label, view.extent(0));
  Kokkos::deep_copy(execution, host, view);
  execution.fence("copy pipeline checkpoint data to host");
  std::vector<Value> values(host.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host(i);
  return values;
}

inline void validate_sorted_registry(const std::vector<int>& modes,
                                     const std::vector<int>& parents,
                                     const std::vector<int>& targets) {
  if (modes.empty() || parents.empty() || targets.empty() ||
      !std::is_sorted(modes.begin(), modes.end()) ||
      !std::is_sorted(parents.begin(), parents.end()) ||
      !std::is_sorted(targets.begin(), targets.end()) ||
      std::adjacent_find(modes.begin(), modes.end()) != modes.end() ||
      std::adjacent_find(parents.begin(), parents.end()) != parents.end() ||
      std::adjacent_find(targets.begin(), targets.end()) != targets.end()) {
    throw std::runtime_error(
        "checkpoint modes and targets must be nonempty, sorted, and unique");
  }
  try {
    const ModeRegistry registry(modes, parents, targets);
    if (!registry.is_closed_under_sharp()) {
      throw std::runtime_error("checkpoint modes are not sharp closed");
    }
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("invalid checkpoint mode registry");
  }
}

inline void validate_metadata(const PipelineCheckpointMetadata& metadata) {
  if ((metadata.version != pipeline_checkpoint_format_version &&
       metadata.version != pipeline_checkpoint_legacy_d42_version) ||
      metadata.shape.fields != point_pipeline_field_count ||
      metadata.shape.modes != metadata.modes.size() ||
      metadata.shape.radial != metadata.radial_coordinates.size() ||
      metadata.shape.theta != metadata.theta_coordinates.size() ||
      metadata.configuration.theta_nodes !=
          static_cast<int>(metadata.shape.theta)) {
    throw std::runtime_error("inconsistent pipeline checkpoint metadata");
  }
  (void)checked_snapshot_value_count(metadata.shape);
  validate_sorted_registry(metadata.modes, metadata.parents, metadata.targets);
  try {
    validate_configuration(metadata.configuration);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("invalid pipeline checkpoint configuration");
  }
  validate_finite(metadata.progress.time, "checkpoint time");
  if (metadata.progress.time < 0.0) {
    throw std::runtime_error("checkpoint time must be nonnegative");
  }
  validate_source_activation_state(metadata.source_activation,
                                   metadata.progress.time);
  for (const double radius : metadata.radial_coordinates) {
    validate_finite(radius, "checkpoint radial coordinate");
  }
  for (const double theta : metadata.theta_coordinates) {
    validate_finite(theta, "checkpoint theta coordinate");
  }
  const bool theta_increasing =
      std::is_sorted(metadata.theta_coordinates.begin(),
                     metadata.theta_coordinates.end());
  const bool theta_decreasing =
      std::is_sorted(metadata.theta_coordinates.rbegin(),
                     metadata.theta_coordinates.rend());
  if (metadata.radial_coordinates.size() <
          radial_minimum_points(
              metadata.configuration.radial_discretization) ||
      !std::is_sorted(metadata.radial_coordinates.begin(),
                      metadata.radial_coordinates.end()) ||
      std::adjacent_find(metadata.radial_coordinates.begin(),
                         metadata.radial_coordinates.end()) !=
          metadata.radial_coordinates.end() ||
      (!theta_increasing && !theta_decreasing) ||
      std::adjacent_find(metadata.theta_coordinates.begin(),
                         metadata.theta_coordinates.end()) !=
          metadata.theta_coordinates.end()) {
    throw std::runtime_error("invalid checkpoint spatial geometry");
  }
  const double spacing =
      (metadata.radial_coordinates.back() -
       metadata.radial_coordinates.front()) /
      static_cast<double>(metadata.radial_coordinates.size() - 1);
  // An optimizing host compiler may reassociate lower+h*i even though the
  // stored endpoint and spacing recover through separate operations here.
  // Permit only two binary64 ulps; the full coordinate vector is still
  // serialized at max_digits10 and compared bit-for-bit with pipeline storage
  // below.
  for (std::size_t radial = 0; radial < metadata.radial_coordinates.size();
       ++radial) {
    const double expected = metadata.radial_coordinates.front() +
                            spacing * static_cast<double>(radial);
    double lower = expected;
    double upper = expected;
    for (int ulp = 0; ulp < 2; ++ulp) {
      lower = std::nextafter(lower,
                             -std::numeric_limits<double>::infinity());
      upper = std::nextafter(upper,
                             std::numeric_limits<double>::infinity());
    }
    if (metadata.radial_coordinates[radial] < lower ||
        metadata.radial_coordinates[radial] > upper) {
      throw std::runtime_error("checkpoint radial grid is not uniform");
    }
  }
}

inline std::string serialize_metadata(
    const PipelineCheckpointMetadata& metadata) {
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "format=teuk-kokkos-pipeline-checkpoint\n"
         << "version=" << metadata.version << '\n'
         << "ordering=mode,field,radial,theta\n"
         << "complex_storage=interleaved_real_imag_float64\n"
         << "byte_order=" << native_byte_order() << '\n'
         << "state_file=" << pipeline_checkpoint_state_file << '\n'
         << "modes=" << metadata.shape.modes << '\n'
         << "fields=" << metadata.shape.fields << '\n'
         << "radial=" << metadata.shape.radial << '\n'
         << "theta=" << metadata.shape.theta << '\n'
         << "m_values=";
  write_list(output, metadata.modes);
  output << "target_values=";
  write_list(output, metadata.targets);
  output << "parent_values=";
  write_list(output, metadata.parents);
  output << "radial_coordinates=";
  write_list(output, metadata.radial_coordinates);
  output << "theta_coordinates=";
  write_list(output, metadata.theta_coordinates);
  output << "mass=" << metadata.configuration.background.mass << '\n'
         << "spin=" << metadata.configuration.background.spin << '\n'
         << "compactification_length="
         << metadata.configuration.background.compactification_length << '\n'
         << "ell_max_first=" << metadata.configuration.ell_max_first << '\n'
         << "ell_max_second=" << metadata.configuration.ell_max_second << '\n'
         << "theta_nodes=" << metadata.configuration.theta_nodes << '\n'
         << "reduction_damping="
         << metadata.configuration.reduction_damping << '\n'
         << "dissipation=" << metadata.configuration.dissipation << '\n'
         << "reduction=" << reduction_name(metadata.configuration.reduction)
         << '\n'
         << "radial_discretization="
         << radial_discretization_name(
                metadata.configuration.radial_discretization)
         << '\n'
         << "time_step=" << metadata.configuration.time_step << '\n'
         << "source_mode="
         << source_mode_name(metadata.configuration.source_policy.mode) << '\n'
         << "source_start_time="
         << metadata.configuration.source_policy.source_start_time << '\n'
         << "source_normalized_constraint_tolerance="
         << metadata.configuration.source_policy.normalized_constraint_tolerance
         << '\n'
         << "source_required_consecutive_passes="
         << metadata.configuration.source_policy.required_consecutive_passes
         << '\n'
         << "source_active=" << (metadata.source_activation.active ? 1 : 0)
         << '\n'
         << "source_activation_time="
         << metadata.source_activation.activation_time << '\n'
         << "source_consecutive_passes="
         << metadata.source_activation.consecutive_passes << '\n'
         << "source_last_eligibility_time="
         << metadata.source_activation.last_eligibility_time << '\n'
         << "time=" << metadata.progress.time << '\n'
         << "step=" << metadata.progress.step << '\n'
         << "state_checksum_fnv1a64=" << metadata.state_checksum << '\n';
  if (!output) throw std::runtime_error("failed serializing checkpoint metadata");
  return output.str();
}

inline std::vector<std::uint8_t> serialize_state_bytes(
    const std::vector<Complex>& values) {
  std::vector<std::uint8_t> bytes(values.size() * 2U * sizeof(double));
  std::size_t offset = 0;
  for (const Complex value : values) {
    const double parts[2]{value.real(), value.imag()};
    std::memcpy(bytes.data() + offset, parts, sizeof(parts));
    offset += sizeof(parts);
  }
  return bytes;
}

inline void write_exact_bytes(const std::filesystem::path& path,
                              const std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot open checkpoint byte output");
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("failed writing checkpoint bytes");
}

inline std::vector<std::uint8_t> read_exact_bytes(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open checkpoint content");
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("cannot size checkpoint content");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  if (!input) throw std::runtime_error("failed reading checkpoint content");
  return bytes;
}

inline std::vector<Complex> parse_state_bytes(
    const std::span<const std::uint8_t> bytes,
    const std::size_t expected_values) {
  if (bytes.size() != expected_values * 2U * sizeof(double)) {
    throw std::runtime_error("binary snapshot size does not match metadata");
  }
  std::vector<Complex> values(expected_values);
  std::size_t offset = 0;
  for (Complex& value : values) {
    double parts[2]{};
    std::memcpy(parts, bytes.data() + offset, sizeof(parts));
    value = Complex(parts[0], parts[1]);
    offset += sizeof(parts);
  }
  return values;
}

class TemporaryDirectoryGuard {
 public:
  explicit TemporaryDirectoryGuard(std::filesystem::path path)
      : path_(std::move(path)) {}
  TemporaryDirectoryGuard(const TemporaryDirectoryGuard&) = delete;
  TemporaryDirectoryGuard& operator=(const TemporaryDirectoryGuard&) = delete;
  ~TemporaryDirectoryGuard() {
    if (!released_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }
  void release() { released_ = true; }

 private:
  std::filesystem::path path_;
  bool released_ = false;
};

inline std::filesystem::path create_temporary_directory(
    const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> sequence{0};
  std::filesystem::path parent = destination.parent_path();
  if (parent.empty()) parent = ".";
  std::filesystem::create_directories(parent);
  const auto tick = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  for (std::uint64_t attempt = 0; attempt < 100; ++attempt) {
    const std::filesystem::path candidate =
        parent /
        (destination.filename().string() + ".tmp." + std::to_string(tick) +
         "." + std::to_string(sequence.fetch_add(1)));
    std::error_code error;
    if (std::filesystem::create_directory(candidate, error)) return candidate;
    if (error && error != std::errc::file_exists) {
      throw std::runtime_error("cannot create temporary checkpoint directory");
    }
  }
  throw std::runtime_error("cannot reserve temporary checkpoint directory");
}

inline std::map<std::string, std::string> read_entries(std::istream& input) {
  std::map<std::string, std::string> entries;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t delimiter = line.find('=');
    if (delimiter == std::string::npos || delimiter == 0) {
      throw std::runtime_error("malformed checkpoint metadata line");
    }
    const std::string key = line.substr(0, delimiter);
    if (!entries.emplace(key, line.substr(delimiter + 1)).second) {
      throw std::runtime_error("duplicate checkpoint metadata key: " + key);
    }
  }
  if (!input.eof()) throw std::runtime_error("failed while reading checkpoint metadata");

  std::set<std::string> expected{
      "format",          "version",          "ordering",
      "complex_storage", "byte_order",       "state_file",
      "modes",           "fields",           "radial",
      "theta",           "m_values",         "parent_values",
      "target_values",
      "radial_coordinates", "theta_coordinates", "mass",
      "spin",            "compactification_length", "ell_max_first",
      "ell_max_second",
      "theta_nodes",     "reduction_damping", "dissipation",
      "reduction",       "radial_discretization", "time_step", "source_mode",
      "source_start_time", "source_normalized_constraint_tolerance",
      "source_required_consecutive_passes", "source_active",
      "source_activation_time", "source_consecutive_passes",
      "source_last_eligibility_time", "time",
      "step",            "state_checksum_fnv1a64"};
  const std::uint64_t version = parse_number<std::uint64_t>(entries, "version");
  if (version == pipeline_checkpoint_legacy_d42_version) {
    expected.erase("radial_discretization");
  }
  if (entries.size() != expected.size()) {
    throw std::runtime_error("checkpoint metadata key set does not match format");
  }
  for (const auto& [key, value] : entries) {
    (void)value;
    if (!expected.contains(key)) {
      throw std::runtime_error("unknown checkpoint metadata key: " + key);
    }
  }
  return entries;
}

inline std::map<std::string, std::string> read_entries(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open checkpoint metadata");
  return read_entries(input);
}

inline std::map<std::string, std::string> read_entries(
    const std::span<const std::uint8_t> bytes) {
  const std::string text(reinterpret_cast<const char*>(bytes.data()),
                         bytes.size());
  std::istringstream input(text);
  return read_entries(input);
}

inline void validate_storage_geometry(
    const ExecutionSpace& execution, const SpatialPipelineStorage& storage,
    const PipelineCheckpointMetadata& metadata) {
  if (!same_shape(storage.snapshot_shape(), metadata.shape)) {
    throw std::runtime_error("checkpoint shape does not match pipeline");
  }
  const auto modes =
      copy_rank_one_to_host(execution, storage.modes(), "checkpoint_modes");
  const auto radius =
      copy_rank_one_to_host(execution, storage.radius(), "checkpoint_radius");
  const auto theta =
      copy_rank_one_to_host(execution, storage.theta(), "checkpoint_theta");
  if (modes != metadata.modes || radius != metadata.radial_coordinates ||
      theta != metadata.theta_coordinates) {
    throw std::runtime_error("checkpoint geometry does not match pipeline");
  }
}

}  // namespace pipeline_checkpoint_detail

inline PipelineCheckpointMetadata parse_pipeline_checkpoint_metadata(
    const std::map<std::string, std::string>& entries) {
  using namespace pipeline_checkpoint_detail;
  if (require_text(entries, "format") !=
          "teuk-kokkos-pipeline-checkpoint" ||
      require_text(entries, "ordering") != "mode,field,radial,theta" ||
      require_text(entries, "complex_storage") !=
          "interleaved_real_imag_float64" ||
      require_text(entries, "byte_order") != native_byte_order() ||
      require_text(entries, "state_file") != pipeline_checkpoint_state_file) {
    throw std::runtime_error("unsupported pipeline checkpoint format");
  }

  PipelineCheckpointMetadata metadata;
  metadata.version = parse_number<std::uint64_t>(entries, "version");
  metadata.shape.modes = parse_number<std::uint64_t>(entries, "modes");
  metadata.shape.fields = parse_number<std::uint64_t>(entries, "fields");
  metadata.shape.radial = parse_number<std::uint64_t>(entries, "radial");
  metadata.shape.theta = parse_number<std::uint64_t>(entries, "theta");
  metadata.modes = parse_list<int>(entries, "m_values");
  metadata.parents = parse_list<int>(entries, "parent_values");
  metadata.targets = parse_list<int>(entries, "target_values");
  metadata.radial_coordinates =
      parse_list<double>(entries, "radial_coordinates");
  metadata.theta_coordinates =
      parse_list<double>(entries, "theta_coordinates");
  metadata.configuration.background.mass =
      parse_number<double>(entries, "mass");
  metadata.configuration.background.spin =
      parse_number<double>(entries, "spin");
  metadata.configuration.background.compactification_length =
      parse_number<double>(entries, "compactification_length");
  metadata.configuration.ell_max_first =
      parse_number<int>(entries, "ell_max_first");
  metadata.configuration.ell_max_second =
      parse_number<int>(entries, "ell_max_second");
  metadata.configuration.theta_nodes =
      parse_number<int>(entries, "theta_nodes");
  metadata.configuration.reduction_damping =
      parse_number<double>(entries, "reduction_damping");
  metadata.configuration.dissipation =
      parse_number<double>(entries, "dissipation");
  metadata.configuration.reduction =
      parse_reduction(require_text(entries, "reduction"));
  metadata.configuration.radial_discretization =
      metadata.version == pipeline_checkpoint_legacy_d42_version
          ? RadialDiscretization::D42
          : parse_radial_discretization(
                require_text(entries, "radial_discretization"));
  metadata.configuration.time_step =
      parse_number<double>(entries, "time_step");
  metadata.configuration.source_policy.mode =
      parse_source_mode(require_text(entries, "source_mode"));
  metadata.configuration.source_policy.source_start_time =
      parse_number<double>(entries, "source_start_time");
  metadata.configuration.source_policy.normalized_constraint_tolerance =
      parse_number<double>(entries, "source_normalized_constraint_tolerance");
  metadata.configuration.source_policy.required_consecutive_passes =
      parse_number<int>(entries, "source_required_consecutive_passes");
  const int source_active = parse_number<int>(entries, "source_active");
  if (source_active != 0 && source_active != 1) {
    throw std::runtime_error("invalid checkpoint source-active flag");
  }
  metadata.source_activation.active = source_active != 0;
  metadata.source_activation.activation_time =
      parse_number<double>(entries, "source_activation_time");
  metadata.source_activation.consecutive_passes =
      parse_number<int>(entries, "source_consecutive_passes");
  metadata.source_activation.last_eligibility_time =
      parse_number<double>(entries, "source_last_eligibility_time");
  metadata.progress.time = parse_number<double>(entries, "time");
  metadata.progress.step = parse_number<std::uint64_t>(entries, "step");
  metadata.state_checksum =
      parse_number<std::uint64_t>(entries, "state_checksum_fnv1a64");
  validate_metadata(metadata);
  return metadata;
}

inline PipelineCheckpointMetadata read_pipeline_checkpoint_metadata(
    const std::filesystem::path& directory) {
  using namespace pipeline_checkpoint_detail;
  return parse_pipeline_checkpoint_metadata(
      read_entries(directory / pipeline_checkpoint_metadata_file));
}

struct VerifiedPipelineCheckpointLoad {
  PipelineCheckpointMetadata metadata;
  VerifiedPrimaryCheckpointReceipt receipt;
};

// The receipt constructor is intentionally reachable only from the two
// primary checkpoint codec operations below.  An arbitrary digest or byte
// buffer can be inspected, but it cannot be promoted into verified replay
// authority without passing the writer/loader's complete validation path.
class PipelineCheckpointCodec {
 private:
  static VerifiedPrimaryCheckpointReceipt issue_receipt(
      const std::span<const std::uint8_t> metadata,
      const std::span<const std::uint8_t> state, const double time,
      const std::uint64_t step, const std::size_t state_value_count) {
    const auto content =
        make_primary_checkpoint_content_identity(metadata, state).digest;
    const auto state_digest =
        checkpoint_identity_detail::checkpoint_state_digest(state);
    return {content, state_digest, time, step, state_value_count};
  }

  friend VerifiedPrimaryCheckpointReceipt write_pipeline_checkpoint(
      const ExecutionSpace&, const std::filesystem::path&,
      const SpatialPipeline&, const ModeRegistry&,
      const PipelineCheckpointConfiguration&,
      PipelineCheckpointProgress);
  friend VerifiedPipelineCheckpointLoad load_pipeline_checkpoint_verified(
      const ExecutionSpace&, const std::filesystem::path&, SpatialPipeline&,
      const ModeRegistry&, const PipelineCheckpointConfiguration&);
};

// Writes a new checkpoint directory. The complete binary and metadata are
// first written beneath a unique sibling directory and then published with one
// directory rename. Existing destinations are rejected rather than replaced.
inline VerifiedPrimaryCheckpointReceipt write_pipeline_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& directory,
    const SpatialPipeline& pipeline, const ModeRegistry& registry,
    const PipelineCheckpointConfiguration& configuration,
    const PipelineCheckpointProgress progress) {
  using namespace pipeline_checkpoint_detail;
  if (directory.empty() || directory.filename().empty() ||
      directory.filename() == "." || directory.filename() == "..") {
    throw std::invalid_argument("checkpoint destination must name a directory");
  }
  if (std::filesystem::exists(directory)) {
    throw std::runtime_error("checkpoint destination already exists");
  }
  validate_configuration(configuration);
  if (!same_source_policy(pipeline.source_policy(),
                          configuration.source_policy)) {
    throw std::invalid_argument(
        "checkpoint source policy does not match the pipeline");
  }
  if (pipeline.radial_discretization() !=
      configuration.radial_discretization) {
    throw std::invalid_argument(
        "checkpoint radial discretization does not match the pipeline");
  }
  validate_finite(progress.time, "checkpoint time");
  if (progress.time < 0.0) {
    throw std::invalid_argument("checkpoint time must be nonnegative");
  }

  PipelineCheckpointMetadata metadata;
  metadata.shape = pipeline.storage().snapshot_shape();
  metadata.modes = registry.modes();
  metadata.parents = registry.parents();
  metadata.targets = registry.targets();
  metadata.configuration = configuration;
  metadata.progress = progress;
  metadata.source_activation = pipeline.source_activation_state();
  metadata.radial_coordinates = copy_rank_one_to_host(
      execution, pipeline.storage().radius(), "checkpoint_write_radius");
  metadata.theta_coordinates = copy_rank_one_to_host(
      execution, pipeline.storage().theta(), "checkpoint_write_theta");
  validate_metadata(metadata);
  validate_storage_geometry(execution, pipeline.storage(), metadata);

  Kokkos::View<Complex*, Kokkos::HostSpace> host_state(
      "checkpoint_write_state", pipeline.storage().value_count());
  Kokkos::deep_copy(execution, host_state, pipeline.storage().flat_state());
  execution.fence("copy pipeline checkpoint state to host");
  std::vector<Complex> values(host_state.extent(0));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = host_state(i);
  metadata.state_checksum = checksum(values);

  const std::string metadata_text = serialize_metadata(metadata);
  const std::vector<std::uint8_t> state_bytes =
      serialize_state_bytes(values);
  const auto receipt = PipelineCheckpointCodec::issue_receipt(
      std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(metadata_text.data()),
          metadata_text.size()),
      state_bytes, progress.time, progress.step, values.size());

  const std::filesystem::path temporary =
      create_temporary_directory(directory);
  TemporaryDirectoryGuard cleanup(temporary);
  write_exact_bytes(temporary / pipeline_checkpoint_state_file, state_bytes);
  write_exact_bytes(
      temporary / pipeline_checkpoint_metadata_file,
      std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(metadata_text.data()),
          metadata_text.size()));
  std::filesystem::rename(temporary, directory);
  cleanup.release();
  return receipt;
}

// Read-only inspection only.  Unlike VerifiedPrimaryCheckpointReceipt this
// does not validate metadata, checksum, geometry, or state semantics.
inline PrimaryCheckpointContentIdentity
inspect_pipeline_checkpoint_content_identity(
    const std::filesystem::path& directory) {
  return make_primary_checkpoint_content_identity(
      directory / pipeline_checkpoint_metadata_file,
      directory / pipeline_checkpoint_state_file);
}

// All metadata, geometry, byte count, and checksum checks complete before the
// caller's pipeline state is changed.
inline VerifiedPipelineCheckpointLoad load_pipeline_checkpoint_verified(
    const ExecutionSpace& execution, const std::filesystem::path& directory,
    SpatialPipeline& pipeline, const ModeRegistry& expected_registry,
    const PipelineCheckpointConfiguration& expected_configuration) {
  using namespace pipeline_checkpoint_detail;
  const std::vector<std::uint8_t> metadata_bytes = read_exact_bytes(
      directory / pipeline_checkpoint_metadata_file);
  const PipelineCheckpointMetadata metadata =
      parse_pipeline_checkpoint_metadata(read_entries(metadata_bytes));
  validate_configuration(expected_configuration);
  if (!same_source_policy(pipeline.source_policy(),
                          expected_configuration.source_policy)) {
    throw std::runtime_error(
        "checkpoint source policy does not match the caller pipeline");
  }
  if (pipeline.radial_discretization() !=
      expected_configuration.radial_discretization) {
    throw std::runtime_error(
        "checkpoint radial discretization does not match caller pipeline");
  }
  if (metadata.modes != expected_registry.modes() ||
      metadata.parents != expected_registry.parents() ||
      metadata.targets != expected_registry.targets()) {
    throw std::runtime_error("checkpoint mode registry does not match caller");
  }
  if (!same_configuration(metadata.configuration, expected_configuration)) {
    throw std::runtime_error("checkpoint numerical configuration does not match caller");
  }
  validate_storage_geometry(execution, pipeline.storage(), metadata);
  const std::size_t count = checked_snapshot_value_count(metadata.shape);
  const std::vector<std::uint8_t> state_bytes =
      read_exact_bytes(directory / pipeline_checkpoint_state_file);
  const std::vector<Complex> values = parse_state_bytes(state_bytes, count);
  if (checksum(values) != metadata.state_checksum) {
    throw std::runtime_error("pipeline checkpoint state checksum mismatch");
  }

  using ConstHostState =
      Kokkos::View<const Complex****, Kokkos::LayoutRight, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  ConstHostState checkpoint_state(
      values.data(), metadata.shape.modes, metadata.shape.fields,
      metadata.shape.radial, metadata.shape.theta);
  constexpr double restart_band_tolerance = 5.0e-11;
  const auto band_report = measure_pipeline_state_off_band(
      checkpoint_state, expected_registry,
      {expected_configuration.ell_max_first,
       expected_configuration.ell_max_second});
  if (!pipeline_state_is_bandlimited(band_report,
                                     restart_band_tolerance)) {
    throw std::runtime_error(
        "pipeline checkpoint contains meaningful off-band angular content");
  }
  const auto receipt = PipelineCheckpointCodec::issue_receipt(
      metadata_bytes, state_bytes, metadata.progress.time,
      metadata.progress.step, values.size());

  Kokkos::View<Complex*, Kokkos::HostSpace> host("checkpoint_read_state",
                                                  values.size());
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(execution, pipeline.storage().flat_state(), host);
  pipeline.restore_source_activation(execution, metadata.source_activation);
  execution.fence("restore pipeline checkpoint state");
  return {metadata, receipt};
}

inline PipelineCheckpointMetadata load_pipeline_checkpoint(
    const ExecutionSpace& execution, const std::filesystem::path& directory,
    SpatialPipeline& pipeline, const ModeRegistry& expected_registry,
    const PipelineCheckpointConfiguration& expected_configuration) {
  return load_pipeline_checkpoint_verified(execution, directory, pipeline,
                                           expected_registry,
                                           expected_configuration)
      .metadata;
}

template <class PrimaryStateView>
inline void require_primary_state_matches_receipt(
    const ExecutionSpace& execution, const PrimaryStateView& primary,
    const VerifiedPrimaryCheckpointReceipt& receipt) {
  static_assert(PrimaryStateView::rank == 1,
                "verified primary checkpoint state must be rank one");
  if (primary.extent(0) != receipt.state_value_count()) {
    throw std::invalid_argument(
        "primary state extent does not match checkpoint receipt");
  }
  Kokkos::View<Complex*, Kokkos::HostSpace> host(
      "verify_primary_checkpoint_receipt", primary.extent(0));
  Kokkos::deep_copy(execution, host, primary);
  execution.fence("verify primary checkpoint state content");
  std::vector<Complex> values(host.extent(0));
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = host(index);
  }
  const auto state_bytes = pipeline_checkpoint_detail::serialize_state_bytes(values);
  if (checkpoint_identity_detail::checkpoint_state_digest(state_bytes) !=
      receipt.state_digest()) {
    throw std::invalid_argument(
        "primary state bytes do not match checkpoint receipt");
  }
}

}  // namespace teuk
