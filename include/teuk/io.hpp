#pragma once

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "teuk/types.hpp"

namespace teuk {

struct SnapshotShape {
  std::uint64_t modes = 0;
  std::uint64_t fields = 0;
  std::uint64_t radial = 0;
  std::uint64_t theta = 0;
};

struct RunProvenance {
  double time = 0.0;
  std::uint64_t step = 0;
  double time_step = 0.0;
  double mass = 1.0;
  double spin = 0.0;
  double compactification_length = 1.0;
  std::string equation_bundle_sha256 = "unknown";
  std::string git_commit = "unknown";
  std::string device = "unknown";
};

struct SnapshotMetadata {
  SnapshotShape shape;
  std::vector<int> modes;
  std::string backend;
  RunProvenance provenance;
};

namespace detail {

inline void validate_metadata_text(const std::string_view name,
                                   const std::string_view value) {
  if (value.find_first_of("\r\n") != std::string_view::npos) {
    throw std::invalid_argument(std::string(name) + " contains a newline");
  }
}

template <class Value>
Value parse_metadata_number(const std::map<std::string, std::string>& entries,
                            const char* key) {
  const auto iterator = entries.find(key);
  if (iterator == entries.end()) {
    throw std::runtime_error(std::string("missing metadata key: ") + key);
  }
  Value value{};
  const std::string& text = iterator->second;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                      value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::runtime_error(std::string("invalid metadata value for ") + key);
  }
  return value;
}

inline const std::string& require_metadata_text(
    const std::map<std::string, std::string>& entries, const char* key) {
  const auto iterator = entries.find(key);
  if (iterator == entries.end()) {
    throw std::runtime_error(std::string("missing metadata key: ") + key);
  }
  return iterator->second;
}

}  // namespace detail

inline void write_metadata(const std::filesystem::path& directory,
                           const SnapshotShape shape,
                           const std::vector<int>& modes,
                           const std::string& backend,
                           const RunProvenance& provenance = {}) {
  if (shape.modes != modes.size()) {
    throw std::invalid_argument("snapshot mode count does not match m_values");
  }
  detail::validate_metadata_text("backend", backend);
  detail::validate_metadata_text("equation_bundle_sha256",
                                 provenance.equation_bundle_sha256);
  detail::validate_metadata_text("git_commit", provenance.git_commit);
  detail::validate_metadata_text("device", provenance.device);
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / "metadata.txt");
  if (!output) throw std::runtime_error("cannot open metadata output");
  output << "format=teuk-kokkos-complex128-v2\n"
         << "ordering=mode,field,radial,theta\n"
         << "endianness=native\n"
         << "complex_storage=interleaved_real_imag_float64\n"
         << "modes=" << shape.modes << '\n'
         << "fields=" << shape.fields << '\n'
         << "radial=" << shape.radial << '\n'
         << "theta=" << shape.theta << '\n'
         << "backend=" << backend << '\n'
         << "m_values=";
  for (std::size_t i = 0; i < modes.size(); ++i) {
    if (i != 0) output << ',';
    output << modes[i];
  }
  output << '\n'
         << "time=" << provenance.time << '\n'
         << "step=" << provenance.step << '\n'
         << "time_step=" << provenance.time_step << '\n'
         << "mass=" << provenance.mass << '\n'
         << "spin=" << provenance.spin << '\n'
         << "compactification_length="
         << provenance.compactification_length << '\n'
         << "equation_bundle_sha256=" << provenance.equation_bundle_sha256
         << '\n'
         << "git_commit=" << provenance.git_commit << '\n'
         << "device=" << provenance.device << '\n';
  if (!output) throw std::runtime_error("failed while writing metadata");
}

inline void write_complex_snapshot(const std::filesystem::path& path,
                                   const std::vector<Complex>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot open binary snapshot output");
  for (const Complex value : values) {
    const double parts[2] = {value.real(), value.imag()};
    output.write(reinterpret_cast<const char*>(parts),
                 static_cast<std::streamsize>(sizeof(parts)));
  }
  if (!output) throw std::runtime_error("failed while writing binary snapshot");
}

inline SnapshotMetadata read_metadata(const std::filesystem::path& directory) {
  std::ifstream input(directory / "metadata.txt");
  if (!input) throw std::runtime_error("cannot open metadata input");
  std::map<std::string, std::string> entries;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t delimiter = line.find('=');
    if (delimiter == std::string::npos || delimiter == 0) {
      throw std::runtime_error("malformed metadata line");
    }
    const std::string key = line.substr(0, delimiter);
    if (!entries.emplace(key, line.substr(delimiter + 1)).second) {
      throw std::runtime_error("duplicate metadata key: " + key);
    }
  }
  if (!input.eof()) throw std::runtime_error("failed while reading metadata");
  if (detail::require_metadata_text(entries, "format") !=
      "teuk-kokkos-complex128-v2") {
    throw std::runtime_error("unsupported snapshot metadata format");
  }
  if (detail::require_metadata_text(entries, "ordering") !=
          "mode,field,radial,theta" ||
      detail::require_metadata_text(entries, "complex_storage") !=
          "interleaved_real_imag_float64") {
    throw std::runtime_error("unsupported snapshot storage convention");
  }

  SnapshotMetadata metadata;
  metadata.shape.modes =
      detail::parse_metadata_number<std::uint64_t>(entries, "modes");
  metadata.shape.fields =
      detail::parse_metadata_number<std::uint64_t>(entries, "fields");
  metadata.shape.radial =
      detail::parse_metadata_number<std::uint64_t>(entries, "radial");
  metadata.shape.theta =
      detail::parse_metadata_number<std::uint64_t>(entries, "theta");
  metadata.backend = detail::require_metadata_text(entries, "backend");

  const std::string& mode_text =
      detail::require_metadata_text(entries, "m_values");
  std::size_t begin = 0;
  while (begin < mode_text.size()) {
    const std::size_t end = mode_text.find(',', begin);
    const std::string_view token(mode_text.data() + begin,
                                 (end == std::string::npos ? mode_text.size()
                                                          : end) - begin);
    int mode = 0;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), mode);
    if (token.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size()) {
      throw std::runtime_error("invalid m_values metadata");
    }
    metadata.modes.push_back(mode);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (metadata.modes.size() != metadata.shape.modes) {
    throw std::runtime_error("snapshot mode count does not match m_values");
  }

  metadata.provenance.time =
      detail::parse_metadata_number<double>(entries, "time");
  metadata.provenance.step =
      detail::parse_metadata_number<std::uint64_t>(entries, "step");
  metadata.provenance.time_step =
      detail::parse_metadata_number<double>(entries, "time_step");
  metadata.provenance.mass =
      detail::parse_metadata_number<double>(entries, "mass");
  metadata.provenance.spin =
      detail::parse_metadata_number<double>(entries, "spin");
  metadata.provenance.compactification_length =
      detail::parse_metadata_number<double>(entries,
                                            "compactification_length");
  metadata.provenance.equation_bundle_sha256 =
      detail::require_metadata_text(entries, "equation_bundle_sha256");
  metadata.provenance.git_commit =
      detail::require_metadata_text(entries, "git_commit");
  metadata.provenance.device =
      detail::require_metadata_text(entries, "device");
  return metadata;
}

inline std::vector<Complex> read_complex_snapshot(
    const std::filesystem::path& path, const std::size_t expected_values) {
  const std::uintmax_t expected_bytes =
      2 * sizeof(double) * static_cast<std::uintmax_t>(expected_values);
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) != expected_bytes) {
    throw std::runtime_error("binary snapshot size does not match metadata");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open binary snapshot input");
  std::vector<Complex> values(expected_values);
  for (Complex& value : values) {
    double parts[2]{};
    input.read(reinterpret_cast<char*>(parts),
               static_cast<std::streamsize>(sizeof(parts)));
    value = Complex(parts[0], parts[1]);
  }
  if (!input) throw std::runtime_error("failed while reading binary snapshot");
  return values;
}

}  // namespace teuk
