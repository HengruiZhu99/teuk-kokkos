#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace teuk {

// A deliberately tiny first-order jet used only for stage-local source
// tangents. It carries a value and its coordinate-time derivative and applies
// the ordinary chain/product/quotient rules. This is not a general AD system.
template <class T>
struct Jet1 {
  T value{};
  T dt{};

  KOKKOS_INLINE_FUNCTION Jet1() = default;
  KOKKOS_INLINE_FUNCTION Jet1(const T& value_in, const T& dt_in)
      : value(value_in), dt(dt_in) {}
  KOKKOS_INLINE_FUNCTION explicit Jet1(const T& value_in)
      : value(value_in), dt{} {}

  KOKKOS_INLINE_FUNCTION Jet1& operator+=(const Jet1& other) {
    value += other.value;
    dt += other.dt;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION Jet1& operator-=(const Jet1& other) {
    value -= other.value;
    dt -= other.dt;
    return *this;
  }
};

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator+(Jet1<T> left,
                                          const Jet1<T>& right) {
  left += right;
  return left;
}

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator-(Jet1<T> left,
                                          const Jet1<T>& right) {
  left -= right;
  return left;
}

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator-(const Jet1<T>& input) {
  return {-input.value, -input.dt};
}

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator*(const Jet1<T>& left,
                                          const Jet1<T>& right) {
  return {left.value * right.value,
          left.dt * right.value + left.value * right.dt};
}

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator/(const Jet1<T>& numerator,
                                          const Jet1<T>& denominator) {
  const T denominator_squared = denominator.value * denominator.value;
  return {numerator.value / denominator.value,
          (numerator.dt * denominator.value -
           numerator.value * denominator.dt) /
              denominator_squared};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator+(const Jet1<T>& left,
                                          const S& right) {
  return {left.value + T(right), left.dt};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator+(const S& left,
                                          const Jet1<T>& right) {
  return right + left;
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator-(const Jet1<T>& left,
                                          const S& right) {
  return {left.value - T(right), left.dt};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator-(const S& left,
                                          const Jet1<T>& right) {
  return {T(left) - right.value, -right.dt};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator*(const Jet1<T>& left,
                                          const S& right) {
  const T factor(right);
  return {left.value * factor, left.dt * factor};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator*(const S& left,
                                          const Jet1<T>& right) {
  return right * left;
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator/(const Jet1<T>& left,
                                          const S& right) {
  const T divisor(right);
  return {left.value / divisor, left.dt / divisor};
}

template <class T, class S>
  requires std::is_convertible_v<S, T>
KOKKOS_INLINE_FUNCTION Jet1<T> operator/(const S& left,
                                          const Jet1<T>& right) {
  return Jet1<T>(T(left)) / right;
}

template <class T>
KOKKOS_INLINE_FUNCTION Jet1<T> jet_conj(const Jet1<T>& input) {
  using Kokkos::conj;
  return {conj(input.value), conj(input.dt)};
}

}  // namespace teuk

