#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

#include "teuk/types.hpp"

namespace teuk {

// Common-RK-stage regular curvature fields consumed by the live spin +2
// source composition.  The order is part of the adapter and checkpoint
// contracts; do not reorder it without a format/version change.
enum class Plus2TransportedCurvatureComponent : std::size_t {
  Z0 = 0,
  Z1 = 1,
  Z0T = 2,
  Z1T = 3,
  Z0TT = 4,
  Z1TT = 5,
  Count = 6,
};

// Common-stage derivative slots consumed alongside transported curvature by
// the live source.  They live in this neutral adapter header so composition
// code does not depend on the heavyweight transport implementation.
enum class Plus2BianchiDerivativeComponent : std::size_t {
  CapitalDelta4Z1 = 0,
  CapitalDelta4Z1T = 1,
  EthPrime4Z1 = 2,
  EthPrime4Z1T = 3,
  CapitalDelta5Z0 = 4,
  CapitalDelta5Z0T = 5,
  Eth5Z0 = 6,
  Eth5Z0T = 7,
  Count = 8,
};

using Plus2LiveStampView =
    Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2LiveConstStampView =
    Kokkos::View<const std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2TransportedCurvatureStorageView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2TransportedCurvatureView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiDerivativeView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiConstDerivativeView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;

struct Plus2TransportedCurvatureStage {
  Plus2TransportedCurvatureView fields;
  Plus2LiveConstStampView stamps;
};

struct Plus2BianchiDerivativeStage {
  Plus2BianchiConstDerivativeView fields;
  Plus2LiveConstStampView stamps;
};

}  // namespace teuk
