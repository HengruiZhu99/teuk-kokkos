#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "teuk/types.hpp"

namespace teuk {

// Normalized radial Taylor coefficients: coefficient[k]=partial_R^k f/k!.
// This convention turns products into a small convolution and quotients into
// a triangular solve.  It is a local algebra only; it owns no storage and
// applies no finite-difference or evolution operator.
template <std::size_t Degree, class Scalar = Complex>
struct RouteBRadialTaylorJet {
  Kokkos::Array<Scalar, Degree + 1> coefficient{};

  KOKKOS_INLINE_FUNCTION Scalar& operator[](const std::size_t order) {
    return coefficient[order];
  }
  KOKKOS_INLINE_FUNCTION const Scalar& operator[](
      const std::size_t order) const {
    return coefficient[order];
  }

  KOKKOS_INLINE_FUNCTION static RouteBRadialTaylorJet constant(
      const Scalar& value) {
    RouteBRadialTaylorJet result;
    result[0] = value;
    return result;
  }

  template <std::size_t NewDegree>
  KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<NewDegree, Scalar> truncate()
      const {
    static_assert(NewDegree <= Degree);
    RouteBRadialTaylorJet<NewDegree, Scalar> result;
    for (std::size_t order = 0; order <= NewDegree; ++order) {
      result[order] = coefficient[order];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet& operator+=(
      const RouteBRadialTaylorJet& other) {
    for (std::size_t order = 0; order <= Degree; ++order) {
      coefficient[order] += other[order];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet& operator-=(
      const RouteBRadialTaylorJet& other) {
    for (std::size_t order = 0; order <= Degree; ++order) {
      coefficient[order] -= other[order];
    }
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION constexpr double routeb_factorial(
    const std::size_t order) {
  double result = 1.0;
  for (std::size_t factor = 2; factor <= order; ++factor) result *= factor;
  return result;
}

template <class Scalar, std::size_t Count>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Count - 1, Scalar>
routeb_radial_jet_from_derivatives(
    const Kokkos::Array<Scalar, Count>& derivatives) {
  static_assert(Count > 0);
  RouteBRadialTaylorJet<Count - 1, Scalar> result;
  for (std::size_t order = 0; order < Count; ++order) {
    result[order] = derivatives[order] / routeb_factorial(order);
  }
  return result;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION Scalar routeb_radial_jet_derivative(
    const RouteBRadialTaylorJet<Degree, Scalar>& jet,
    const std::size_t order) {
  return routeb_factorial(order) * jet[order];
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator+(
    RouteBRadialTaylorJet<Degree, Scalar> lhs,
    const RouteBRadialTaylorJet<Degree, Scalar>& rhs) {
  lhs += rhs;
  return lhs;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator-(
    RouteBRadialTaylorJet<Degree, Scalar> lhs,
    const RouteBRadialTaylorJet<Degree, Scalar>& rhs) {
  lhs -= rhs;
  return lhs;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator-(
    const RouteBRadialTaylorJet<Degree, Scalar>& value) {
  RouteBRadialTaylorJet<Degree, Scalar> result;
  for (std::size_t order = 0; order <= Degree; ++order) {
    result[order] = -value[order];
  }
  return result;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator*(
    const RouteBRadialTaylorJet<Degree, Scalar>& lhs,
    const RouteBRadialTaylorJet<Degree, Scalar>& rhs) {
  RouteBRadialTaylorJet<Degree, Scalar> result;
  for (std::size_t order = 0; order <= Degree; ++order) {
    for (std::size_t left = 0; left <= order; ++left) {
      result[order] += lhs[left] * rhs[order - left];
    }
  }
  return result;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar>
routeb_radial_jet_reciprocal(
    const RouteBRadialTaylorJet<Degree, Scalar>& denominator) {
  RouteBRadialTaylorJet<Degree, Scalar> result;
  result[0] = Scalar(1.0) / denominator[0];
  for (std::size_t order = 1; order <= Degree; ++order) {
    Scalar accumulated{};
    for (std::size_t right = 1; right <= order; ++right) {
      accumulated += denominator[right] * result[order - right];
    }
    result[order] = -accumulated / denominator[0];
  }
  return result;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator/(
    const RouteBRadialTaylorJet<Degree, Scalar>& numerator,
    const RouteBRadialTaylorJet<Degree, Scalar>& denominator) {
  return numerator * routeb_radial_jet_reciprocal(denominator);
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator*(
    const Scalar& scale, RouteBRadialTaylorJet<Degree, Scalar> jet) {
  for (std::size_t order = 0; order <= Degree; ++order) jet[order] *= scale;
  return jet;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator*(
    RouteBRadialTaylorJet<Degree, Scalar> jet, const Scalar& scale) {
  return scale * jet;
}

template <std::size_t Degree, class Scalar>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Scalar> operator/(
    RouteBRadialTaylorJet<Degree, Scalar> jet, const Scalar& scale) {
  for (std::size_t order = 0; order <= Degree; ++order) jet[order] /= scale;
  return jet;
}

template <std::size_t Degree, class Scalar>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree - 1, Scalar>
routeb_radial_jet_derivative(
    const RouteBRadialTaylorJet<Degree, Scalar>& jet) {
  RouteBRadialTaylorJet<Degree - 1, Scalar> result;
  for (std::size_t order = 0; order < Degree; ++order) {
    result[order] = static_cast<double>(order + 1) * jet[order + 1];
  }
  return result;
}

}  // namespace teuk
