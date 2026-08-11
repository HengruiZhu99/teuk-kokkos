#pragma once

#include <cstddef>

#include "teuk/types.hpp"

namespace teuk {

enum class TeukolskyField : std::size_t { P = 0, Q = 1, Psi = 2, Count = 3 };

enum class ReconstructionField : std::size_t {
  G = 0,
  Lambda = 1,
  H = 2,
  B = 3,
  Pi = 4,
  C = 5,
  U = 6,
  Count = 7
};

// Logical ordering is (mode, field, radial, theta), with theta contiguous.
using FieldView = Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;

struct FieldMetadata {
  const char* name;
  int spin;
  int boost;
  int radial_falloff;
};

inline constexpr FieldMetadata reconstruction_metadata[] = {
    {"G", -1, -1, 2}, {"Lambda", -2, -1, 1}, {"H", 0, 0, 3},
    {"B", -2, 0, 1},  {"Pi", -1, 0, 2},      {"C", -1, 1, 2},
    {"U", 0, 1, 3}};

}  // namespace teuk

