#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/types.hpp"

namespace teuk {

struct SnapshotShape {
  std::uint64_t modes = 0;
  std::uint64_t fields = 0;
  std::uint64_t radial = 0;
  std::uint64_t theta = 0;
};

inline void write_metadata(const std::filesystem::path& directory,
                           const SnapshotShape shape,
                           const std::vector<int>& modes,
                           const std::string& backend) {
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / "metadata.txt");
  if (!output) throw std::runtime_error("cannot open metadata output");
  output << "format=teuk-kokkos-complex128-v1\n"
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
  output << '\n';
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

}  // namespace teuk

