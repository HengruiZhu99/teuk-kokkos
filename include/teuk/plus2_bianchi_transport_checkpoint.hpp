#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
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

#include "teuk/plus2_bianchi_transport.hpp"

namespace teuk {

inline constexpr std::uint32_t plus2_bianchi_checkpoint_version = 1;
inline constexpr const char* plus2_bianchi_checkpoint_schema =
    "teuk.plus2-bianchi-transport-checkpoint";
inline constexpr const char* plus2_bianchi_checkpoint_scaling =
    "Psi0=R^5*Z0;Psi1=R^4*Z1";
inline constexpr const char* plus2_bianchi_checkpoint_storage =
    "LayoutRight(mode,field,radial,theta);field-order=(Z0,Z1)";

struct Plus2BianchiCheckpointProgress {
  double time = 0.0;
  std::uint64_t step = 0;
};

struct Plus2BianchiCheckpointProvenance {
  std::string git_commit;
  int runtime_config_schema_version = 0;
  std::string boundary_evidence_id;
};

struct Plus2BianchiCheckpointMetadata {
  std::string schema = plus2_bianchi_checkpoint_schema;
  std::uint32_t version = plus2_bianchi_checkpoint_version;
  std::string byte_order;
  std::string floating_point_format = "IEEE-754-binary64";
  std::string complex_component_order = "real-then-imag";
  std::string storage_order = plus2_bianchi_checkpoint_storage;
  std::string scaling = plus2_bianchi_checkpoint_scaling;
  std::vector<int> modes;
  int ell_max = 0;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
  double lower_radius = 0.0;
  double upper_radius = 0.0;
  KerrParameters background;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  std::string initialization_evidence_id;
  std::string boundary_evidence_id;
  bool independently_qualified_scri_coefficients = false;
  std::string git_commit;
  int runtime_config_schema_version = 0;
  Plus2BianchiCheckpointProgress progress;
  std::uint64_t last_generation = 0;
  std::uint64_t state_checksum = 0;
};

struct Plus2BianchiCheckpointExpectations {
  std::vector<int> modes;
  int ell_max = 0;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
  double lower_radius = 0.0;
  double upper_radius = 0.0;
  KerrParameters background;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  std::string initialization_evidence_id;
  std::string boundary_evidence_id;
  bool independently_qualified_scri_coefficients = false;
  std::string git_commit;
  int runtime_config_schema_version = 0;
};

namespace plus2_bianchi_checkpoint_detail {

inline constexpr std::array<char, 8> magic{'T', 'E', 'U', 'K', 'B', 'I', 'A',
                                           '1'};

inline const char* native_byte_order() {
  if constexpr (std::endian::native == std::endian::little) return "little";
  if constexpr (std::endian::native == std::endian::big) return "big";
  throw std::runtime_error("mixed-endian Bianchi checkpoint unsupported");
}

inline void require_git_commit(const std::string& commit) {
  const bool valid =
      commit.size() >= 7 &&
      std::all_of(commit.begin(), commit.end(), [](const unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
      });
  if (!valid) {
    throw std::invalid_argument(
        "Bianchi checkpoint requires an explicit Git commit");
  }
}

inline void validate_registry(const std::vector<int>& modes,
                              const int ell_max) {
  if (modes.empty() || ell_max < 2 ||
      !std::is_sorted(modes.begin(), modes.end()) ||
      std::adjacent_find(modes.begin(), modes.end()) != modes.end()) {
    throw std::invalid_argument("invalid Bianchi checkpoint mode registry");
  }
  for (const int mode : modes) {
    if (std::abs(mode) > ell_max ||
        !std::binary_search(modes.begin(), modes.end(), -mode)) {
      throw std::invalid_argument(
          "Bianchi checkpoint modes must be in-band and sharp closed");
    }
  }
}

inline std::size_t checked_state_count(const std::size_t modes,
                                       const std::size_t radial,
                                       const std::size_t theta) {
  constexpr std::size_t fields =
      static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
  std::size_t result = modes;
  for (const std::size_t factor : {fields, radial, theta}) {
    if (factor != 0 &&
        result > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::overflow_error("Bianchi checkpoint state size overflow");
    }
    result *= factor;
  }
  return result;
}

inline void validate_metadata(const Plus2BianchiCheckpointMetadata& m) {
  if (m.schema != plus2_bianchi_checkpoint_schema ||
      m.version != plus2_bianchi_checkpoint_version ||
      m.byte_order != native_byte_order() ||
      m.floating_point_format != "IEEE-754-binary64" ||
      m.complex_component_order != "real-then-imag" ||
      m.storage_order != plus2_bianchi_checkpoint_storage ||
      m.scaling != plus2_bianchi_checkpoint_scaling ||
      m.radial_count < radial_minimum_points(m.radial_discretization) ||
      m.theta_count == 0 || !(m.upper_radius > m.lower_radius) ||
      !std::isfinite(m.lower_radius) || !std::isfinite(m.upper_radius) ||
      m.lower_radius < 0.0 || !std::isfinite(m.background.mass) ||
      !std::isfinite(m.background.spin) ||
      !std::isfinite(m.background.compactification_length) ||
      m.background.mass <= 0.0 ||
      std::abs(m.background.spin) > m.background.mass ||
      m.background.compactification_length <= 0.0 ||
      m.initialization_evidence_id.empty() ||
      m.boundary_evidence_id.empty() ||
      (m.lower_radius == 0.0 &&
       !m.independently_qualified_scri_coefficients) ||
      !std::isfinite(m.progress.time) || m.progress.time < 0.0 ||
      m.last_generation == std::numeric_limits<std::uint64_t>::max() ||
      sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559 ||
      std::numeric_limits<double>::digits != 53) {
    throw std::invalid_argument(
        "invalid Bianchi checkpoint representation or geometry");
  }
  validate_registry(m.modes, m.ell_max);
  require_git_commit(m.git_commit);
  if (m.runtime_config_schema_version <= 0) {
    throw std::invalid_argument(
        "Bianchi checkpoint runtime schema must be positive");
  }
  static_cast<void>(radial_discretization_name(m.radial_discretization));
  static_cast<void>(checked_state_count(m.modes.size(), m.radial_count,
                                        m.theta_count));
}

inline std::uint64_t checksum(const std::vector<Complex>& values) {
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

inline void validate_state_values(const std::vector<Complex>& values,
                                  const std::size_t expected_count) {
  if (values.size() != expected_count) {
    throw std::runtime_error("Bianchi checkpoint state count mismatch");
  }
  for (const Complex value : values) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      throw std::runtime_error("nonfinite Bianchi checkpoint state");
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
    throw std::runtime_error(std::string("truncated Bianchi checkpoint at ") +
                             label);
  }
  return value;
}

inline void write_string(std::ostream& output, const std::string& value) {
  write_scalar(output, static_cast<std::uint64_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline std::string read_string(std::istream& input, const char* label) {
  constexpr std::uint64_t maximum = 4096;
  const auto size = read_scalar<std::uint64_t>(input, label);
  if (size > maximum) {
    throw std::runtime_error("unreasonable Bianchi checkpoint string");
  }
  std::string value(static_cast<std::size_t>(size), '\0');
  input.read(value.data(), static_cast<std::streamsize>(value.size()));
  if (!input) {
    throw std::runtime_error(std::string("truncated Bianchi checkpoint at ") +
                             label);
  }
  return value;
}

inline void write_modes(std::ostream& output, const std::vector<int>& modes) {
  write_scalar(output, static_cast<std::uint64_t>(modes.size()));
  for (const int mode : modes) {
    write_scalar(output, static_cast<std::int32_t>(mode));
  }
}

inline std::vector<int> read_modes(std::istream& input) {
  constexpr std::uint64_t maximum = 1ULL << 20;
  const auto count = read_scalar<std::uint64_t>(input, "mode count");
  if (count > maximum) {
    throw std::runtime_error("unreasonable Bianchi checkpoint mode count");
  }
  std::vector<int> modes(static_cast<std::size_t>(count));
  for (int& mode : modes) {
    mode = static_cast<int>(read_scalar<std::int32_t>(input, "mode"));
  }
  return modes;
}

inline void write_payload(std::ostream& output,
                          const Plus2BianchiCheckpointMetadata& m,
                          const std::vector<Complex>& values) {
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  write_string(output, m.schema);
  write_scalar(output, m.version);
  write_string(output, m.byte_order);
  write_string(output, m.floating_point_format);
  write_string(output, m.complex_component_order);
  write_string(output, m.storage_order);
  write_string(output, m.scaling);
  write_modes(output, m.modes);
  write_scalar(output, static_cast<std::int32_t>(m.ell_max));
  write_scalar(output, static_cast<std::uint64_t>(m.radial_count));
  write_scalar(output, static_cast<std::uint64_t>(m.theta_count));
  write_scalar(output, m.lower_radius);
  write_scalar(output, m.upper_radius);
  write_scalar(output, m.background.mass);
  write_scalar(output, m.background.spin);
  write_scalar(output, m.background.compactification_length);
  write_string(output, radial_discretization_name(m.radial_discretization));
  write_string(output, m.initialization_evidence_id);
  write_string(output, m.boundary_evidence_id);
  write_scalar(output, static_cast<std::uint8_t>(
                           m.independently_qualified_scri_coefficients));
  write_string(output, m.git_commit);
  write_scalar(output,
               static_cast<std::int32_t>(m.runtime_config_schema_version));
  write_scalar(output, m.progress.time);
  write_scalar(output, m.progress.step);
  write_scalar(output, m.last_generation);
  write_scalar(output, m.state_checksum);
  write_scalar(output, static_cast<std::uint64_t>(values.size()));
  for (const Complex value : values) {
    write_scalar(output, value.real());
    write_scalar(output, value.imag());
  }
}

inline std::pair<Plus2BianchiCheckpointMetadata, std::vector<Complex>>
read_payload(std::istream& input) {
  std::array<char, magic.size()> input_magic{};
  input.read(input_magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || input_magic != magic) {
    throw std::runtime_error("invalid Bianchi checkpoint magic");
  }
  Plus2BianchiCheckpointMetadata m;
  m.schema = read_string(input, "schema");
  m.version = read_scalar<std::uint32_t>(input, "version");
  m.byte_order = read_string(input, "byte order");
  m.floating_point_format = read_string(input, "floating point");
  m.complex_component_order = read_string(input, "complex order");
  m.storage_order = read_string(input, "storage order");
  m.scaling = read_string(input, "scaling");
  m.modes = read_modes(input);
  m.ell_max = static_cast<int>(read_scalar<std::int32_t>(input, "ell max"));
  m.radial_count = static_cast<std::size_t>(
      read_scalar<std::uint64_t>(input, "radial count"));
  m.theta_count = static_cast<std::size_t>(
      read_scalar<std::uint64_t>(input, "theta count"));
  m.lower_radius = read_scalar<double>(input, "lower radius");
  m.upper_radius = read_scalar<double>(input, "upper radius");
  m.background.mass = read_scalar<double>(input, "mass");
  m.background.spin = read_scalar<double>(input, "spin");
  m.background.compactification_length =
      read_scalar<double>(input, "compactification length");
  m.radial_discretization =
      parse_radial_discretization(read_string(input, "radial scheme"));
  m.initialization_evidence_id =
      read_string(input, "initialization evidence");
  m.boundary_evidence_id = read_string(input, "boundary evidence");
  const auto scri = read_scalar<std::uint8_t>(input, "scri evidence");
  if (scri > 1) throw std::runtime_error("invalid scri evidence flag");
  m.independently_qualified_scri_coefficients = scri != 0;
  m.git_commit = read_string(input, "git commit");
  m.runtime_config_schema_version = static_cast<int>(
      read_scalar<std::int32_t>(input, "runtime schema"));
  m.progress.time = read_scalar<double>(input, "time");
  m.progress.step = read_scalar<std::uint64_t>(input, "step");
  m.last_generation = read_scalar<std::uint64_t>(input, "generation");
  m.state_checksum = read_scalar<std::uint64_t>(input, "checksum");
  const auto count = read_scalar<std::uint64_t>(input, "state count");
  validate_metadata(m);
  const auto expected =
      checked_state_count(m.modes.size(), m.radial_count, m.theta_count);
  if (count != expected)
    throw std::runtime_error("Bianchi checkpoint state count mismatch");
  std::vector<Complex> values(expected);
  for (Complex& value : values) {
    const double real = read_scalar<double>(input, "state real");
    const double imaginary = read_scalar<double>(input, "state imaginary");
    value = Complex(real, imaginary);
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("trailing Bianchi checkpoint data");
  }
  validate_state_values(values, expected);
  return {std::move(m), std::move(values)};
}

inline void require_match(const Plus2BianchiCheckpointMetadata& m,
                          const Plus2BianchiCheckpointExpectations& e) {
  if (m.modes != e.modes || m.ell_max != e.ell_max ||
      m.radial_count != e.radial_count || m.theta_count != e.theta_count ||
      m.lower_radius != e.lower_radius || m.upper_radius != e.upper_radius ||
      m.background.mass != e.background.mass ||
      m.background.spin != e.background.spin ||
      m.background.compactification_length !=
          e.background.compactification_length ||
      m.radial_discretization != e.radial_discretization ||
      m.initialization_evidence_id != e.initialization_evidence_id ||
      m.boundary_evidence_id != e.boundary_evidence_id ||
      m.independently_qualified_scri_coefficients !=
          e.independently_qualified_scri_coefficients ||
      m.git_commit != e.git_commit ||
      m.runtime_config_schema_version != e.runtime_config_schema_version) {
    throw std::runtime_error(
        "Bianchi checkpoint does not match geometry, methods, or evidence");
  }
}

}  // namespace plus2_bianchi_checkpoint_detail

template <class ExecSpace>
Plus2BianchiCheckpointExpectations plus2_bianchi_checkpoint_expectations(
    const Plus2BianchiTransport<ExecSpace>& transport,
    const Plus2BianchiCheckpointProvenance& provenance,
    const std::string& initialization_evidence_id,
    const bool independently_qualified_scri_coefficients) {
  return {transport.registry().modes(),
          transport.ell_max(),
          transport.radial_count(),
          transport.theta_count(),
          transport.radial_grid().lower_radius(),
          transport.radial_grid().upper_radius(),
          transport.parameters(),
          transport.radial_discretization(),
          initialization_evidence_id,
          provenance.boundary_evidence_id,
          independently_qualified_scri_coefficients,
          provenance.git_commit,
          provenance.runtime_config_schema_version};
}

template <class ExecSpace>
Plus2BianchiCheckpointMetadata save_plus2_bianchi_transport_checkpoint(
    const ExecSpace& execution, const std::filesystem::path& path,
    const Plus2BianchiTransport<ExecSpace>& transport,
    const Plus2BianchiCheckpointProgress& progress,
    const Plus2BianchiCheckpointProvenance& provenance) {
  using namespace plus2_bianchi_checkpoint_detail;
  if (!transport.initialized() || path.empty() ||
      std::filesystem::exists(path) ||
      provenance.boundary_evidence_id != transport.boundary_evidence_id()) {
    throw std::invalid_argument(
        "Bianchi checkpoint target/state is invalid or already exists");
  }
  Plus2BianchiCheckpointMetadata metadata;
  metadata.byte_order = native_byte_order();
  metadata.modes = transport.registry().modes();
  metadata.ell_max = transport.ell_max();
  metadata.radial_count = transport.radial_count();
  metadata.theta_count = transport.theta_count();
  metadata.lower_radius = transport.radial_grid().lower_radius();
  metadata.upper_radius = transport.radial_grid().upper_radius();
  metadata.background = transport.parameters();
  metadata.radial_discretization = transport.radial_discretization();
  metadata.initialization_evidence_id =
      transport.initialization_evidence_id();
  metadata.boundary_evidence_id = provenance.boundary_evidence_id;
  metadata.independently_qualified_scri_coefficients =
      transport.scri_coefficients_qualified();
  metadata.git_commit = provenance.git_commit;
  metadata.runtime_config_schema_version =
      provenance.runtime_config_schema_version;
  metadata.progress = progress;
  metadata.last_generation = transport.last_generation();

  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, transport.state());
  execution.fence("copy Bianchi checkpoint state to host");
  std::vector<Complex> values(host.data(), host.data() + host.size());
  validate_state_values(
      values, checked_state_count(metadata.modes.size(), metadata.radial_count,
                                  metadata.theta_count));
  metadata.state_checksum = checksum(values);
  validate_metadata(metadata);

  const auto temporary = path.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error("Bianchi checkpoint temporary path exists");
  }
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create Bianchi checkpoint");
    write_payload(output, metadata, values);
    output.close();
    if (!output) throw std::runtime_error("failed writing Bianchi checkpoint");
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  return metadata;
}

template <class ExecSpace>
Plus2BianchiCheckpointMetadata load_plus2_bianchi_transport_checkpoint(
    const ExecSpace& execution, const std::filesystem::path& path,
    Plus2BianchiTransport<ExecSpace>& transport,
    const Plus2BianchiCheckpointExpectations& expected) {
  using namespace plus2_bianchi_checkpoint_detail;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open Bianchi checkpoint");
  auto [metadata, values] = read_payload(input);
  require_match(metadata, expected);
  if (checksum(values) != metadata.state_checksum) {
    throw std::runtime_error("Bianchi checkpoint state checksum mismatch");
  }

  Kokkos::View<Complex****, Kokkos::LayoutRight, Kokkos::HostSpace> host(
      "Bianchi checkpoint restore", metadata.modes.size(),
      static_cast<std::size_t>(Plus2BianchiStateComponent::Count),
      metadata.radial_count, metadata.theta_count);
  std::copy(values.begin(), values.end(), host.data());
  transport.restore_checkpoint_state(
      execution, host,
      {metadata.initialization_evidence_id, true, true,
       metadata.independently_qualified_scri_coefficients,
       metadata.boundary_evidence_id},
      metadata.last_generation);
  execution.fence("restore Bianchi checkpoint state");
  return metadata;
}

}  // namespace teuk
