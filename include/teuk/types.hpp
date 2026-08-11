#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace teuk {

using Real = double;
using Complex = Kokkos::complex<Real>;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <class DataType>
using View = Kokkos::View<DataType, MemorySpace>;

KOKKOS_INLINE_FUNCTION
constexpr Real square(const Real value) { return value * value; }

}  // namespace teuk

