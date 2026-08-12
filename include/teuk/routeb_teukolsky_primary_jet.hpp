#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/fields.hpp"
#include "teuk/grid.hpp"
#include "teuk/routeb_fornberg.hpp"
#include "teuk/routeb_radial_taylor_jet.hpp"
#include "teuk/teukolsky.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr std::size_t routeb_teukolsky_level_count = 5;

template <std::size_t Degree>
struct RouteBTeukolskyStateJet {
  RouteBRadialTaylorJet<Degree, Complex> P;
  RouteBRadialTaylorJet<Degree, Complex> Q;
  RouteBRadialTaylorJet<Degree, Complex> psi;
};

template <std::size_t Degree>
struct RouteBTeukolskyCoefficientJets {
  RouteBRadialTaylorJet<Degree, Complex> time;
  RouteBRadialTaylorJet<Degree, Complex> radial_advection;
  RouteBRadialTaylorJet<Degree, Complex> radial_principal;
  RouteBRadialTaylorJet<Degree, Complex> definition;
  RouteBRadialTaylorJet<Degree, Complex> q;
  RouteBRadialTaylorJet<Degree, Complex> psi;
};

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Complex>
routeb_complex_constant(const Complex value) {
  return RouteBRadialTaylorJet<Degree, Complex>::constant(value);
}

// The same stationary Kerr coefficients as teukolsky_coefficients, evaluated
// over a normalized radial Taylor jet. This is explicit product/quotient
// algebra, not differentiation of sampled coefficient fields.
template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBTeukolskyCoefficientJets<Degree>
routeb_teukolsky_coefficient_jets(const TeukolskyParameters& parameters,
                                  const double radius,
                                  const double theta) {
  using Jet = RouteBRadialTaylorJet<Degree, Complex>;
  const double mass = parameters.mass;
  const double spin = parameters.spin;
  const double length = parameters.compactification_length;
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const double spin2 = spin * spin;
  const double s = static_cast<double>(parameters.spin_weight);
  const double m = static_cast<double>(parameters.azimuthal_mode);
  const Complex imaginary_unit(0.0, 1.0);
  Jet R = Jet::constant(Complex(radius, 0.0));
  if constexpr (Degree > 0) R[1] = Complex(1.0, 0.0);
  const Jet R2 = R * R;
  const Jet one = Jet::constant(Complex(1.0, 0.0));

  RouteBTeukolskyCoefficientJets<Degree> result;
  result.time =
      Complex(8.0 * mass, 0.0) *
          (Complex(2.0 * mass, 0.0) * one -
           Complex(spin2 / length2, 0.0) * R) *
          (one + Complex(2.0 * mass / length2, 0.0) * R) -
      Complex(spin2 * Kokkos::sin(theta) * Kokkos::sin(theta), 0.0) * one;
  result.radial_advection =
      Complex(length2, 0.0) * one -
      Complex((8.0 * mass * mass - spin2) / length2, 0.0) * R2 +
      Complex(4.0 * spin2 * mass / length4, 0.0) * (R2 * R);
  result.radial_principal =
      Complex(1.0 / length4, 0.0) * R2 *
      (Complex(length4, 0.0) * one -
       Complex(2.0 * length2 * mass, 0.0) * R +
       Complex(spin2, 0.0) * R2);
  result.definition =
      Complex(0.0, 2.0 * spin * m) *
          (one + Complex(4.0 * mass / length2, 0.0) * R) +
      Complex(4.0 * mass, 0.0) *
          (Complex(-s, 0.0) * one +
           Complex((2.0 + s) * 2.0 * mass / length2, 0.0) * R -
           Complex(3.0 * spin2 / length4, 0.0) * R2) -
      Complex(2.0 * spin2 / length2, 0.0) * R +
      Complex(0.0, 2.0 * s * spin * Kokkos::cos(theta)) * one;
  result.q =
      Complex(2.0, 0.0) * R *
          (Complex(1.0 + s, 0.0) * one -
           Complex((3.0 + s) * mass / length2, 0.0) * R +
           Complex(2.0 * spin2 / length4, 0.0) * R2) -
      Complex(0.0, 2.0 * spin * m / length2) * R2;
  result.psi =
      Complex(-2.0, 0.0) * R *
          (Complex((1.0 + s) * mass / length2, 0.0) * one -
           Complex(spin2 / length4, 0.0) * R) -
      Complex(0.0, 2.0 * spin * m / length2) * R;
  return result;
}

template <std::size_t Degree>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBTeukolskyStateJet<Degree - 1>
routeb_teukolsky_primary_jet_step(
    const TeukolskyParameters& parameters, const double radius,
    const double theta, const RouteBTeukolskyStateJet<Degree>& state,
    const RouteBRadialTaylorJet<Degree - 1, Complex>& angular_laplacian) {
  const auto coefficients =
      routeb_teukolsky_coefficient_jets<Degree>(parameters, radius, theta);
  const auto psi_velocity =
      (state.P + Complex(2.0, 0.0) * coefficients.radial_advection * state.Q -
       coefficients.definition * state.psi) /
      coefficients.time;
  const auto derivative_q = routeb_radial_jet_derivative(state.Q);
  const auto derivative_psi = routeb_radial_jet_derivative(state.psi);
  RouteBTeukolskyStateJet<Degree - 1> result;
  result.P = coefficients.radial_principal.template truncate<Degree - 1>() *
                 derivative_q +
             coefficients.q.template truncate<Degree - 1>() *
                 state.Q.template truncate<Degree - 1>() +
             coefficients.psi.template truncate<Degree - 1>() *
                 state.psi.template truncate<Degree - 1>() +
             angular_laplacian;
  result.Q = routeb_radial_jet_derivative(psi_velocity) -
             Complex(parameters.reduction_damping, 0.0) *
                 (state.Q.template truncate<Degree - 1>() - derivative_psi);
  result.psi = psi_velocity.template truncate<Degree - 1>();
  return result;
}

namespace routeb_teukolsky_detail {

using Stride3 = std::array<std::size_t, 3>;
using Stride4 = std::array<std::size_t, 4>;
using Stride5 = std::array<std::size_t, 5>;

KOKKOS_INLINE_FUNCTION std::size_t index3(
    const std::size_t a, const std::size_t b, const std::size_t c,
    const Stride3& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2];
}
KOKKOS_INLINE_FUNCTION std::size_t index4(
    const std::size_t a, const std::size_t b, const std::size_t c,
    const std::size_t d, const Stride4& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3];
}
KOKKOS_INLINE_FUNCTION std::size_t index5(
    const std::size_t a, const std::size_t b, const std::size_t c,
    const std::size_t d, const std::size_t e, const Stride5& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3] +
         e * stride[4];
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<4, Complex> pad_jet(
    const RouteBRadialTaylorJet<Degree, Complex>& source) {
  RouteBRadialTaylorJet<4, Complex> result;
  for (std::size_t order = 0; order <= Degree; ++order) {
    result[order] = source[order];
  }
  return result;
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBTeukolskyStateJet<Degree> load_state(
    const RouteBRadialTaylorJet<4, Complex>* jets, const std::size_t mode,
    const std::size_t radial, const std::size_t theta,
    const Stride5& stride, const std::size_t level) {
  RouteBTeukolskyStateJet<Degree> result;
  result.P = jets[index5(level, mode, 0, radial, theta, stride)]
                 .template truncate<Degree>();
  result.Q = jets[index5(level, mode, 1, radial, theta, stride)]
                 .template truncate<Degree>();
  result.psi = jets[index5(level, mode, 2, radial, theta, stride)]
                   .template truncate<Degree>();
  return result;
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION void store_state_raw(
    const RouteBTeukolskyStateJet<Degree>& state,
    RouteBRadialTaylorJet<4, Complex>* jets, Complex* values,
    const std::size_t level, const std::size_t mode,
    const std::size_t radial, const std::size_t theta,
    const Stride5& jet_stride, const Stride5& value_stride) {
  const RouteBRadialTaylorJet<4, Complex> padded[3] = {
      pad_jet(state.P), pad_jet(state.Q), pad_jet(state.psi)};
  for (std::size_t field = 0; field < 3; ++field) {
    const std::size_t index =
        index5(level, mode, field, radial, theta, value_stride);
    values[index] = padded[field][0];
    jets[index5(level, mode, field, radial, theta, jet_stride)] =
        padded[field];
  }
}

struct ValidateInitialFunctor {
  const Complex* input;
  const std::uint64_t* stamps;
  std::uint8_t* ready;
  Stride4 input_stride;
  Stride3 stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::uint64_t generation;
  const Real* theta_coordinates;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = generation != 0 &&
                 stamps[index3(mode, radial, theta, stamp_stride)] ==
                     generation;
    const Real theta_value = theta_coordinates[theta];
    valid = valid && Kokkos::isfinite(theta_value) && theta_value >= 0.0 &&
            theta_value <= Kokkos::acos(-1.0);
    for (std::size_t field = 0; field < 3; ++field) {
      const Complex value =
          input[index4(mode, field, radial, theta, input_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag());
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct InitializeFunctor {
  const Complex* input;
  const Complex* derivatives;
  RouteBRadialTaylorJet<4, Complex>* jets;
  Complex* values;
  Stride4 input_stride;
  Stride5 derivative_stride;
  Stride5 jet_stride;
  Stride5 value_stride;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    RouteBTeukolskyStateJet<4> state;
    RouteBRadialTaylorJet<4, Complex>* fields[3] = {&state.P, &state.Q,
                                                    &state.psi};
    for (std::size_t field = 0; field < 3; ++field) {
      (*fields[field])[0] =
          input[index4(mode, field, radial, theta, input_stride)];
      for (std::size_t order = 1; order <= 4; ++order) {
        (*fields[field])[order] =
            derivatives[index5(mode, order - 1, field, radial, theta,
                               derivative_stride)] /
            routeb_factorial(order);
      }
    }
    store_state_raw(state, jets, values, 0, mode, radial, theta, jet_stride,
                    value_stride);
  }
};

struct ValidateAngularFunctor {
  const Complex* angular;
  const std::uint64_t* angular_stamps;
  const std::uint64_t* current_stamps;
  std::uint8_t* ready;
  Stride4 angular_stride;
  Stride4 angular_stamp_stride;
  Stride4 current_stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = generation != 0 &&
                 current_stamps[index4(level, mode, radial, theta,
                                       current_stamp_stride)] == generation;
    const std::size_t active_orders = 4 - level;
    for (std::size_t order = 0; order < active_orders; ++order) {
      const Complex value =
          angular[index4(mode, order, radial, theta, angular_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag()) &&
              angular_stamps[index4(mode, order, radial, theta,
                                    angular_stamp_stride)] == generation;
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct AdvanceFunctor {
  const Complex* angular;
  const int* modes;
  const Real* theta_coordinates;
  RouteBRadialTaylorJet<4, Complex>* jets;
  Complex* values;
  Stride4 angular_stride;
  Stride5 jet_stride;
  Stride5 value_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  UniformRadialGrid grid;
  TeukolskyParameters parameters;

  template <std::size_t Degree>
  KOKKOS_INLINE_FUNCTION void apply(const std::size_t mode,
                                    const std::size_t radial,
                                    const std::size_t theta) const {
    auto state = load_state<Degree>(jets, mode, radial, theta, jet_stride,
                                    level);
    RouteBRadialTaylorJet<Degree - 1, Complex> angular_jet;
    for (std::size_t order = 0; order < Degree; ++order) {
      angular_jet[order] =
          angular[index4(mode, order, radial, theta, angular_stride)];
    }
    TeukolskyParameters point_parameters = parameters;
    point_parameters.azimuthal_mode = modes[mode];
    const auto next = routeb_teukolsky_primary_jet_step(
        point_parameters, grid.coordinate(radial), theta_coordinates[theta],
        state, angular_jet);
    store_state_raw(next, jets, values, level + 1, mode, radial, theta,
                    jet_stride, value_stride);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    if (level == 0) apply<4>(mode, radial, theta);
    if (level == 1) apply<3>(mode, radial, theta);
    if (level == 2) apply<2>(mode, radial, theta);
    if (level == 3) apply<1>(mode, radial, theta);
  }
};

struct ValidateLevelJetsFunctor {
  const RouteBRadialTaylorJet<4, Complex>* jets;
  std::uint8_t* ready;
  Stride5 jet_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = 0; field < 3; ++field) {
      const auto jet =
          jets[index5(level, mode, field, radial, theta, jet_stride)];
      for (std::size_t order = 0; order < 5; ++order) {
        valid = valid && Kokkos::isfinite(jet[order].real()) &&
                Kokkos::isfinite(jet[order].imag());
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct FinalizeLevelFunctor {
  RouteBRadialTaylorJet<4, Complex>* jets;
  Complex* values;
  std::uint64_t* stamps;
  Complex* psi_coefficients;
  std::uint64_t* psi_coefficient_stamps;
  Complex* current_coefficients;
  std::uint64_t* current_coefficient_stamps;
  const std::uint8_t* ready;
  Stride5 jet_stride;
  Stride5 value_stride;
  Stride4 stamp_stride;
  Stride4 psi_coefficient_stride;
  Stride4 psi_coefficient_stamp_stride;
  Stride5 current_coefficient_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < 3; ++field) {
      if (!valid) {
        values[index5(level, mode, field, radial, theta, value_stride)] = {};
        jets[index5(level, mode, field, radial, theta, jet_stride)] = {};
      }
      const auto jet =
          jets[index5(level, mode, field, radial, theta, jet_stride)];
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && order < 5 - level;
        current_coefficients[index5(mode, field, order, radial, theta,
                                    current_coefficient_stride)] =
            active ? jet[order] : Complex{};
        current_coefficient_stamps[index5(
            mode, field, order, radial, theta,
            current_coefficient_stride)] = active ? generation : 0;
      }
    }
    stamps[index4(level, mode, radial, theta, stamp_stride)] =
        valid ? generation : 0;
    for (std::size_t order = 0; order < 4; ++order) {
      const bool active = valid && order < 4 - level;
      psi_coefficients[index4(mode, order, radial, theta,
                              psi_coefficient_stride)] =
          active ? jets[index5(level, mode, 2, radial, theta, jet_stride)]
                       [order]
                 : Complex{};
      psi_coefficient_stamps[index4(mode, order, radial, theta,
                                    psi_coefficient_stamp_stride)] =
          active ? generation : 0;
    }
  }
};

struct ValidateProjectedCoefficientsFunctor {
  const Complex* coefficients;
  const std::uint64_t* stamps;
  std::uint8_t* ready;
  Stride5 value_stride;
  Stride5 stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t active_orders;
  std::uint64_t token;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = 0; field < 3; ++field) {
      for (std::size_t order = 0; order < active_orders; ++order) {
        const auto value = coefficients[index5(mode, field, order, radial,
                                                theta, value_stride)];
        valid = valid && Kokkos::isfinite(value.real()) &&
                Kokkos::isfinite(value.imag()) &&
                stamps[index5(mode, field, order, radial, theta,
                              stamp_stride)] == token;
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct AcceptProjectedCoefficientsFunctor {
  const Complex* projected;
  RouteBRadialTaylorJet<4, Complex>* jets;
  Complex* values;
  std::uint64_t* level_stamps;
  Complex* current_coefficients;
  std::uint64_t* current_stamps;
  Complex* psi_coefficients;
  std::uint64_t* psi_stamps;
  const std::uint8_t* ready;
  Stride5 projected_stride;
  Stride5 jet_stride;
  Stride5 value_stride;
  Stride4 level_stamp_stride;
  Stride5 current_stride;
  Stride4 psi_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t active_orders;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < 3; ++field) {
      RouteBRadialTaylorJet<4, Complex> jet{};
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && order < active_orders;
        const Complex value =
            active ? projected[index5(mode, field, order, radial, theta,
                                      projected_stride)]
                   : Complex{};
        jet[order] = value;
        current_coefficients[index5(mode, field, order, radial, theta,
                                    current_stride)] = value;
        current_stamps[index5(mode, field, order, radial, theta,
                              current_stride)] = active ? generation : 0;
        if (field == 2 && order < 4) {
          psi_coefficients[index4(mode, order, radial, theta, psi_stride)] =
              value;
          psi_stamps[index4(mode, order, radial, theta, psi_stride)] =
              active ? generation : 0;
        }
      }
      jets[index5(level, mode, field, radial, theta, jet_stride)] = jet;
      values[index5(level, mode, field, radial, theta, value_stride)] =
          valid ? jet[0] : Complex{};
    }
    level_stamps[index4(level, mode, radial, theta, level_stamp_stride)] =
        valid ? generation : 0;
  }
};

}  // namespace routeb_teukolsky_detail

template <class ExecutionSpace = teuk::ExecutionSpace>
class RouteBTeukolskyPrimaryJetTower {
 public:
  // Queue-order contract: initialize, every coefficient-wise angular launch,
  // and each advance in one generation must use the same ordered execution
  // space instance. These methods intentionally do not fence and therefore
  // cannot establish dependencies between distinct backend queues.
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using value_view =
      Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>;
  using const_value_view = typename value_view::const_type;
  using stamp_view =
      Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, memory_space>;
  using const_stamp_view = typename stamp_view::const_type;
  using jet_type = RouteBRadialTaylorJet<4, Complex>;
  using jet_view = Kokkos::View<jet_type*****, Kokkos::LayoutRight,
                                memory_space>;
  using psi_coefficient_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, memory_space>;
  using psi_coefficient_stamp_view =
      Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, memory_space>;
  using coefficient_view =
      Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>;
  using coefficient_stamp_view =
      Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, memory_space>;

  RouteBTeukolskyPrimaryJetTower(
      const std::size_t mode_count, const UniformRadialGrid& grid,
      const std::size_t theta_count,
      const std::string& label = "routeb_teukolsky_primary_jet")
      : mode_count_(mode_count),
        grid_(grid),
        theta_count_(theta_count),
        values_(label + "_values", routeb_teukolsky_level_count, mode_count,
                3, grid.size(), theta_count),
        stamps_(label + "_stamps", routeb_teukolsky_level_count, mode_count,
                grid.size(), theta_count),
        jets_(label + "_jets", routeb_teukolsky_level_count, mode_count, 3,
              grid.size(), theta_count),
        input_derivatives_(label + "_input_derivatives", mode_count, 4, 3,
                           grid.size(), theta_count),
        psi_coefficients_(label + "_psi_coefficients", mode_count, 4,
                          grid.size(), theta_count),
        psi_coefficient_stamps_(label + "_psi_coefficient_stamps", mode_count,
                                4, grid.size(), theta_count),
        current_coefficients_(label + "_current_coefficients", mode_count, 3,
                              5, grid.size(), theta_count),
        current_coefficient_stamps_(label + "_current_coefficient_stamps",
                                    mode_count, 3, 5, grid.size(),
                                    theta_count),
        modes_(label + "_modes", mode_count),
        theta_(label + "_theta", theta_count),
        ready_(label + "_ready", 1) {
    if (mode_count == 0 || theta_count == 0 ||
        grid.size() < routeb_fornberg_window) {
      throw std::invalid_argument("Route-B Teukolsky tower extents invalid");
    }
  }

  RouteBTeukolskyPrimaryJetTower(
      const RouteBTeukolskyPrimaryJetTower&) = delete;
  RouteBTeukolskyPrimaryJetTower& operator=(
      const RouteBTeukolskyPrimaryJetTower&) = delete;
  RouteBTeukolskyPrimaryJetTower(RouteBTeukolskyPrimaryJetTower&&) = delete;
  RouteBTeukolskyPrimaryJetTower& operator=(
      RouteBTeukolskyPrimaryJetTower&&) = delete;

  [[nodiscard]] const_value_view values() const { return values_; }
  [[nodiscard]] const_stamp_view stamps() const { return stamps_; }
  [[nodiscard]] typename psi_coefficient_view::const_type psi_coefficients()
      const {
    return psi_coefficients_;
  }
  [[nodiscard]] typename psi_coefficient_stamp_view::const_type
  psi_coefficient_stamps() const {
    return psi_coefficient_stamps_;
  }
  [[nodiscard]] typename coefficient_view::const_type current_coefficients()
      const {
    return current_coefficients_;
  }
  [[nodiscard]] typename coefficient_stamp_view::const_type
  current_coefficient_stamps() const {
    return current_coefficient_stamps_;
  }
  [[nodiscard]] std::size_t current_level() const { return current_level_; }
  [[nodiscard]] std::uint64_t generation() const { return generation_; }
  [[nodiscard]] static bool generation_supported(
      const std::uint64_t generation) {
    return generation != 0 && valid_projection_tokens(generation);
  }

  template <class ModeView, class ThetaView, class InputView,
            class InputStampView>
  void initialize(const execution_space& execution,
                  const TeukolskyParameters& parameters,
                  const ModeView& signed_modes, const ThetaView& theta,
                  const InputView& input,
                  const InputStampView& input_stamps,
                  const std::uint64_t generation,
                  const ReductionEvolution reduction,
                  const double dissipation_strength) {
    validate_parameters(parameters, reduction, dissipation_strength);
    if (!generation_supported(generation)) {
      throw std::invalid_argument("Route-B Teukolsky generation is zero");
    }
    validate_initial_views(signed_modes, theta, input, input_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    Kokkos::deep_copy(execution, values_, Complex{});
    Kokkos::deep_copy(execution, stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, jets_, jet_type{});
    Kokkos::deep_copy(execution, psi_coefficients_, Complex{});
    Kokkos::deep_copy(execution, psi_coefficient_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, current_coefficients_, Complex{});
    Kokkos::deep_copy(execution, current_coefficient_stamps_,
                      std::uint64_t{0});
    Kokkos::deep_copy(execution, modes_, signed_modes);
    Kokkos::deep_copy(execution, theta_, theta);
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_teukolsky_validate_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::ValidateInitialFunctor{
            input.data(), input_stamps.data(), ready_.data(),
            strides4(input), strides3(input_stamps), grid_.size(),
            theta_count_, generation, theta_.data()});
    evaluate_routeb_fornberg_derivatives(execution, input,
                                          input_derivatives_, grid_.spacing());
    Kokkos::parallel_for(
        "routeb_teukolsky_initialize_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::InitializeFunctor{
            input.data(), input_derivatives_.data(), jets_.data(),
            values_.data(), strides4(input), strides5(input_derivatives_),
            strides5(jets_), strides5(values_), grid_.size(), theta_count_});
    Kokkos::parallel_for(
        "routeb_teukolsky_validate_h0_jets",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::ValidateLevelJetsFunctor{
            jets_.data(), ready_.data(), strides5(jets_), grid_.size(),
            theta_count_, 0});
    Kokkos::parallel_for(
        "routeb_teukolsky_finalize_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::FinalizeLevelFunctor{
            jets_.data(), values_.data(), stamps_.data(),
            psi_coefficients_.data(), psi_coefficient_stamps_.data(),
            current_coefficients_.data(), current_coefficient_stamps_.data(),
            ready_.data(),
            strides5(jets_), strides5(values_), strides4(stamps_),
            strides4(psi_coefficients_), strides4(psi_coefficient_stamps_),
            strides5(current_coefficients_),
            grid_.size(), theta_count_, 0, generation});
    parameters_ = parameters;
    generation_ = generation;
    current_level_ = 0;
    current_level_projected_ = false;
    initialized_ = true;
  }

  template <class AngularView, class AngularStampView>
  void advance(const execution_space& execution,
               const AngularView& angular_laplacian,
               const AngularStampView& angular_stamps,
               const std::uint64_t generation,
               const ReductionEvolution reduction,
               const double dissipation_strength) {
    if (!initialized_ || current_level_ >= 4 || generation != generation_) {
      throw std::logic_error("Route-B Teukolsky tower stage is unavailable");
    }
    validate_advance_views(angular_laplacian, angular_stamps);
    if (reduction != ReductionEvolution::FreeDamped ||
        dissipation_strength != 0.0) {
      throw std::invalid_argument(
          "Route-B Teukolsky jets require FreeDamped and zero dissipation");
    }
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_teukolsky_validate_angular_level",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::ValidateAngularFunctor{
            angular_laplacian.data(), angular_stamps.data(), stamps_.data(),
            ready_.data(), strides4(angular_laplacian),
            strides4(angular_stamps), strides4(stamps_), grid_.size(),
            theta_count_, current_level_, generation});
    Kokkos::parallel_for(
        "routeb_teukolsky_advance_one_level",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::AdvanceFunctor{
            angular_laplacian.data(), modes_.data(), theta_.data(),
            jets_.data(), values_.data(), strides4(angular_laplacian),
            strides5(jets_), strides5(values_), grid_.size(), theta_count_,
            current_level_, grid_, parameters_});
    Kokkos::parallel_for(
        "routeb_teukolsky_validate_next_jets",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::ValidateLevelJetsFunctor{
            jets_.data(), ready_.data(), strides5(jets_), grid_.size(),
            theta_count_, current_level_ + 1});
    Kokkos::parallel_for(
        "routeb_teukolsky_finalize_next_level",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::FinalizeLevelFunctor{
            jets_.data(), values_.data(), stamps_.data(),
            psi_coefficients_.data(), psi_coefficient_stamps_.data(),
            current_coefficients_.data(), current_coefficient_stamps_.data(),
            ready_.data(),
            strides5(jets_), strides5(values_), strides4(stamps_),
            strides4(psi_coefficients_), strides4(psi_coefficient_stamps_),
            strides5(current_coefficients_),
            grid_.size(), theta_count_, current_level_ + 1, generation});
    ++current_level_;
    current_level_projected_ = false;
  }

  [[nodiscard]] std::uint64_t expected_projection_token() const {
    if (!initialized_) return 0;
    return projection_token(generation_, current_level_);
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_projected_current(
      const execution_space& execution, const ProjectedView& projected,
      const ProjectedStampView& projected_stamps,
      const std::uint64_t generation, const std::uint64_t token) {
    if (!initialized_ || current_level_projected_ || generation != generation_ ||
        token != expected_projection_token()) {
      throw std::logic_error(
          "Route-B Teukolsky projection stage is unavailable");
    }
    validate_projected_views(projected, projected_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t active = 5 - current_level_;
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_teukolsky_validate_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::ValidateProjectedCoefficientsFunctor{
            projected.data(), projected_stamps.data(), ready_.data(),
            strides5(projected), strides5(projected_stamps), grid_.size(),
            theta_count_, active, token});
    Kokkos::parallel_for(
        "routeb_teukolsky_accept_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_teukolsky_detail::AcceptProjectedCoefficientsFunctor{
            projected.data(), jets_.data(), values_.data(), stamps_.data(),
            current_coefficients_.data(), current_coefficient_stamps_.data(),
            psi_coefficients_.data(), psi_coefficient_stamps_.data(),
            ready_.data(), strides5(projected), strides5(jets_),
            strides5(values_), strides4(stamps_),
            strides5(current_coefficients_), strides4(psi_coefficients_),
            grid_.size(), theta_count_, current_level_, active, generation_});
    current_level_projected_ = true;
  }

 private:
  template <class View>
  static routeb_teukolsky_detail::Stride3 strides3(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2)};
  }
  template <class View>
  static routeb_teukolsky_detail::Stride4 strides4(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2), view.stride(3)};
  }
  template <class View>
  static routeb_teukolsky_detail::Stride5 strides5(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2), view.stride(3),
            view.stride(4)};
  }

  static void validate_parameters(const TeukolskyParameters& parameters,
                                  const ReductionEvolution reduction,
                                  const double dissipation_strength) {
    if (reduction != ReductionEvolution::FreeDamped ||
        dissipation_strength != 0.0) {
      throw std::invalid_argument(
          "Route-B Teukolsky jets require FreeDamped and zero dissipation");
    }
    if (!(parameters.compactification_length > 0.0) ||
        !std::isfinite(parameters.compactification_length) ||
        !(parameters.mass > 0.0) || !std::isfinite(parameters.mass) ||
        !std::isfinite(parameters.spin) ||
        std::abs(parameters.spin) > parameters.mass ||
        !std::isfinite(parameters.reduction_damping) ||
        parameters.reduction_damping < 0.0 || parameters.spin_weight != -2) {
      throw std::invalid_argument("Route-B Teukolsky parameters invalid");
    }
  }

  [[nodiscard]] static std::uint64_t projection_token(
      const std::uint64_t generation, const std::size_t level) {
    return generation ^ (0xa24baed4963ee407ULL * (level + 1)) ^
           0x9fb21c651e98df25ULL;
  }

  [[nodiscard]] static bool valid_projection_tokens(
      const std::uint64_t generation) {
    std::array<std::uint64_t, 5> tokens{};
    for (std::size_t level = 0; level < tokens.size(); ++level) {
      tokens[level] = projection_token(generation, level);
      if (tokens[level] == 0) return false;
      for (std::size_t previous = 0; previous < level; ++previous) {
        if (tokens[previous] == tokens[level]) return false;
      }
    }
    return true;
  }

  template <class View>
  bool overlaps_owned(const View& view) const {
    return routeb_fornberg_detail::allocations_overlap(view, values_) ||
           routeb_fornberg_detail::allocations_overlap(view, stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view, jets_) ||
           routeb_fornberg_detail::allocations_overlap(view,
                                                        input_derivatives_) ||
           routeb_fornberg_detail::allocations_overlap(view,
                                                        psi_coefficients_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, psi_coefficient_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, current_coefficients_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, current_coefficient_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view, modes_) ||
           routeb_fornberg_detail::allocations_overlap(view, theta_) ||
           routeb_fornberg_detail::allocations_overlap(view, ready_);
  }

  template <class ModeView, class ThetaView, class InputView,
            class InputStampView>
  void validate_initial_views(const ModeView& modes, const ThetaView& theta,
                              const InputView& input,
                              const InputStampView& input_stamps) const {
    static_assert(ModeView::rank == 1 && ThetaView::rank == 1 &&
                  InputView::rank == 4 && InputStampView::rank == 3);
    static_assert(
        std::is_same_v<typename ModeView::non_const_value_type, int> &&
        std::is_same_v<typename ThetaView::non_const_value_type, Real> &&
        std::is_same_v<typename InputView::non_const_value_type, Complex> &&
        std::is_same_v<typename InputStampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ModeView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ThetaView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename InputView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename InputStampView::memory_space>::accessible);
    if (modes.data() == nullptr || theta.data() == nullptr ||
        input.data() == nullptr || input_stamps.data() == nullptr ||
        modes.extent(0) != mode_count_ || theta.extent(0) != theta_count_ ||
        input.extent(0) != mode_count_ || input.extent(1) != 3 ||
        input.extent(2) != grid_.size() ||
        input.extent(3) != theta_count_ ||
        input_stamps.extent(0) != mode_count_ ||
        input_stamps.extent(1) != grid_.size() ||
        input_stamps.extent(2) != theta_count_ ||
        !routeb_fornberg_detail::has_separated_strides<4>(input) ||
        !routeb_fornberg_detail::has_separated_strides<3>(input_stamps) ||
        !routeb_fornberg_detail::has_separated_strides<1>(modes) ||
        !routeb_fornberg_detail::has_separated_strides<1>(theta) ||
        overlaps_owned(modes) || overlaps_owned(theta) ||
        overlaps_owned(input) || overlaps_owned(input_stamps)) {
      throw std::invalid_argument("Route-B Teukolsky h0 views invalid");
    }
  }

  template <class AngularView, class AngularStampView>
  void validate_advance_views(const AngularView& angular,
                              const AngularStampView& angular_stamps) const {
    static_assert(AngularView::rank == 4 && AngularStampView::rank == 4);
    static_assert(
        std::is_same_v<typename AngularView::non_const_value_type, Complex> &&
        std::is_same_v<typename AngularStampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename AngularView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename AngularStampView::memory_space>::accessible);
    if (angular.data() == nullptr || angular_stamps.data() == nullptr ||
        angular.extent(0) != mode_count_ || angular.extent(1) != 4 ||
        angular.extent(2) != grid_.size() ||
        angular.extent(3) != theta_count_ ||
        angular_stamps.extent(0) != mode_count_ ||
        angular_stamps.extent(1) != 4 ||
        angular_stamps.extent(2) != grid_.size() ||
        angular_stamps.extent(3) != theta_count_ ||
        !routeb_fornberg_detail::has_separated_strides<4>(angular) ||
        !routeb_fornberg_detail::has_separated_strides<4>(angular_stamps) ||
        overlaps_owned(angular) || overlaps_owned(angular_stamps)) {
      throw std::invalid_argument("Route-B Teukolsky angular views invalid");
    }
  }

  template <class ProjectedView, class ProjectedStampView>
  void validate_projected_views(
      const ProjectedView& projected,
      const ProjectedStampView& projected_stamps) const {
    static_assert(ProjectedView::rank == 5 && ProjectedStampView::rank == 5);
    static_assert(
        std::is_same_v<typename ProjectedView::non_const_value_type, Complex> &&
        std::is_same_v<typename ProjectedStampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ProjectedView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ProjectedStampView::memory_space>::accessible);
    const auto valid = [&](const auto& view) {
      return view.data() != nullptr && view.extent(0) == mode_count_ &&
             view.extent(1) == 3 && view.extent(2) == 5 &&
             view.extent(3) == grid_.size() &&
             view.extent(4) == theta_count_ &&
             routeb_fornberg_detail::has_separated_strides<5>(view) &&
             !overlaps_owned(view);
    };
    if (!valid(projected) || !valid(projected_stamps) ||
        routeb_fornberg_detail::allocations_overlap(projected,
                                                     projected_stamps)) {
      throw std::invalid_argument(
          "Route-B Teukolsky projected coefficient views invalid");
    }
  }

  std::size_t mode_count_;
  UniformRadialGrid grid_;
  std::size_t theta_count_;
  value_view values_;
  stamp_view stamps_;
  jet_view jets_;
  Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>
      input_derivatives_;
  psi_coefficient_view psi_coefficients_;
  psi_coefficient_stamp_view psi_coefficient_stamps_;
  coefficient_view current_coefficients_;
  coefficient_stamp_view current_coefficient_stamps_;
  Kokkos::View<int*, memory_space> modes_;
  Kokkos::View<Real*, memory_space> theta_;
  Kokkos::View<std::uint8_t*, memory_space> ready_;
  TeukolskyParameters parameters_;
  std::size_t current_level_ = 0;
  std::uint64_t generation_ = 0;
  bool initialized_ = false;
  bool current_level_projected_ = false;
};

}  // namespace teuk
