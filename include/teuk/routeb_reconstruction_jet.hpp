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
#include "teuk/reconstruction.hpp"
#include "teuk/routeb_fornberg.hpp"
#include "teuk/routeb_radial_taylor_jet.hpp"
#include "teuk/teukolsky.hpp"

namespace teuk {

inline constexpr std::size_t routeb_reconstruction_field_count =
    static_cast<std::size_t>(ReconstructionField::Count);

enum class RouteBReconstructionPass3Angular : std::size_t {
  Eth2C = 0,
  Eth2Pi = 1,
  EthPrime1BSharp = 2,
  EthPrime2CSharp = 3,
  Count = 4
};

template <std::size_t Degree>
struct RouteBReconstructionStateJet {
  RouteBRadialTaylorJet<Degree, Complex> G;
  RouteBRadialTaylorJet<Degree, Complex> Lambda;
  RouteBRadialTaylorJet<Degree, Complex> H;
  RouteBRadialTaylorJet<Degree, Complex> B;
  RouteBRadialTaylorJet<Degree, Complex> Pi;
  RouteBRadialTaylorJet<Degree, Complex> C;
  RouteBRadialTaylorJet<Degree, Complex> U;
};

template <std::size_t Degree>
struct RouteBReconstructionBackgroundJets {
  RouteBRadialTaylorJet<Degree, Complex> mu0;
  RouteBRadialTaylorJet<Degree, Complex> tau0;
  RouteBRadialTaylorJet<Degree, Complex> pi0;
};

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree, Complex>
routeb_reconstruction_conjugate(
    const RouteBRadialTaylorJet<Degree, Complex>& input) {
  RouteBRadialTaylorJet<Degree, Complex> result;
  for (std::size_t order = 0; order <= Degree; ++order) {
    result[order] = Kokkos::conj(input[order]);
  }
  return result;
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBReconstructionBackgroundJets<Degree>
routeb_reconstruction_background_jets(const KerrParameters& parameters,
                                      const double radius,
                                      const double theta) {
  using Jet = RouteBRadialTaylorJet<Degree, Complex>;
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double length4 = length2 * length2;
  const double spin = parameters.spin;
  const double cosine = Kokkos::cos(theta);
  const double sine = Kokkos::sin(theta);
  const double sqrt_two = Kokkos::sqrt(2.0);
  const Complex imaginary(0.0, 1.0);
  Jet R = Jet::constant(Complex(radius, 0.0));
  if constexpr (Degree > 0) R[1] = Complex(1.0, 0.0);
  const Jet one = Jet::constant(Complex(1.0, 0.0));
  const Jet angular_denominator =
      Complex(length2, 0.0) * one - Complex(0.0, spin * cosine) * R;
  const Jet minus_radial_denominator =
      Complex(-length2, 0.0) * one + Complex(0.0, spin * cosine) * R;
  const Jet real_pi_denominator =
      Complex(length4, 0.0) * one +
      Complex(spin * spin * cosine * cosine, 0.0) * (R * R);
  RouteBReconstructionBackgroundJets<Degree> result;
  result.mu0 = one / minus_radial_denominator;
  result.tau0 = Complex(0.0, spin * sine / sqrt_two) * one /
                (angular_denominator * angular_denominator);
  result.pi0 = Complex(0.0, -spin * sine / sqrt_two) * one /
               real_pi_denominator;
  return result;
}

template <std::size_t Degree>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree - 1, Complex>
routeb_reconstruction_time_derivative_jet(
    const RouteBRadialTaylorJet<Degree, Complex>& value,
    const RouteBRadialTaylorJet<Degree - 1, Complex>& delta_rhs,
    const int radial_falloff, const double radius, const double mass,
    const double compactification_length) {
  using OutputJet = RouteBRadialTaylorJet<Degree - 1, Complex>;
  const double length2 = compactification_length * compactification_length;
  OutputJet R = OutputJet::constant(Complex(radius, 0.0));
  if constexpr (Degree > 1) R[1] = Complex(1.0, 0.0);
  const OutputJet one = OutputJet::constant(Complex(1.0, 0.0));
  const auto truncated = value.template truncate<Degree - 1>();
  const auto radial_terms =
      Complex(1.0 / length2, 0.0) * (R * R) *
          routeb_radial_jet_derivative(value) +
      Complex(static_cast<double>(radial_falloff) / length2, 0.0) * R *
          truncated;
  const auto denominator =
      Complex(2.0, 0.0) * one +
      Complex(4.0 * mass / length2, 0.0) * R;
  return (delta_rhs - radial_terms) / denominator;
}

template <std::size_t Degree>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBReconstructionStateJet<Degree - 1>
routeb_reconstruction_pass1_jet(
    const KerrParameters& parameters, const double radius, const double theta,
    const RouteBReconstructionStateJet<Degree>& state,
    const RouteBRadialTaylorJet<Degree - 1, Complex>& psi4,
    const RouteBRadialTaylorJet<Degree - 1, Complex>& eth1_f) {
  const auto background =
      routeb_reconstruction_background_jets<Degree - 1>(parameters, radius,
                                                         theta);
  RouteBRadialTaylorJet<Degree - 1, Complex> R =
      RouteBRadialTaylorJet<Degree - 1, Complex>::constant(
          Complex(radius, 0.0));
  if constexpr (Degree > 1) R[1] = Complex(1.0, 0.0);
  const auto g = state.G.template truncate<Degree - 1>();
  const auto lambda = state.Lambda.template truncate<Degree - 1>();
  const auto delta_g = Complex(-4.0, 0.0) * R * background.mu0 * g +
                       eth1_f - R * background.tau0 * psi4;
  const auto delta_lambda =
      -R * (background.mu0 + routeb_reconstruction_conjugate(background.mu0)) *
          lambda -
      psi4;
  RouteBReconstructionStateJet<Degree - 1> result;
  result.G = routeb_reconstruction_time_derivative_jet(
      state.G, delta_g, 2, radius, parameters.mass,
      parameters.compactification_length);
  result.Lambda = routeb_reconstruction_time_derivative_jet(
      state.Lambda, delta_lambda, 1, radius, parameters.mass,
      parameters.compactification_length);
  return result;
}

template <std::size_t Degree>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBReconstructionStateJet<Degree - 1>
routeb_reconstruction_pass2_jet(
    const KerrParameters& parameters, const double radius, const double theta,
    const RouteBReconstructionStateJet<Degree>& state,
    const RouteBRadialTaylorJet<Degree - 1, Complex>& eth2_g) {
  const auto background =
      routeb_reconstruction_background_jets<Degree - 1>(parameters, radius,
                                                         theta);
  RouteBRadialTaylorJet<Degree - 1, Complex> R =
      RouteBRadialTaylorJet<Degree - 1, Complex>::constant(
          Complex(radius, 0.0));
  if constexpr (Degree > 1) R[1] = Complex(1.0, 0.0);
  const auto R2 = R * R;
  const auto mu_bar = routeb_reconstruction_conjugate(background.mu0);
  const auto pi_bar = routeb_reconstruction_conjugate(background.pi0);
  const auto g = state.G.template truncate<Degree - 1>();
  const auto lambda = state.Lambda.template truncate<Degree - 1>();
  const auto h = state.H.template truncate<Degree - 1>();
  const auto b = state.B.template truncate<Degree - 1>();
  const auto pi = state.Pi.template truncate<Degree - 1>();
  const auto c = state.C.template truncate<Degree - 1>();
  const auto delta_h = Complex(-3.0, 0.0) * R * background.mu0 * h +
                       eth2_g -
                       Complex(2.0, 0.0) * R * background.tau0 * g;
  const auto delta_b = R * (background.mu0 - mu_bar) * b -
                       Complex(2.0, 0.0) * lambda;
  const auto delta_pi =
      -g - R * (pi_bar + background.tau0) * lambda +
      Complex(0.5, 0.0) * R2 * background.mu0 *
          (pi_bar + background.tau0) * b;
  const auto delta_c = -R * mu_bar * c - Complex(2.0, 0.0) * pi -
                       R * background.tau0 * b;
  RouteBReconstructionStateJet<Degree - 1> result;
  result.H = routeb_reconstruction_time_derivative_jet(
      state.H, delta_h, 3, radius, parameters.mass,
      parameters.compactification_length);
  result.B = routeb_reconstruction_time_derivative_jet(
      state.B, delta_b, 1, radius, parameters.mass,
      parameters.compactification_length);
  result.Pi = routeb_reconstruction_time_derivative_jet(
      state.Pi, delta_pi, 2, radius, parameters.mass,
      parameters.compactification_length);
  result.C = routeb_reconstruction_time_derivative_jet(
      state.C, delta_c, 2, radius, parameters.mass,
      parameters.compactification_length);
  return result;
}

template <std::size_t Degree>
  requires(Degree > 0)
KOKKOS_INLINE_FUNCTION RouteBRadialTaylorJet<Degree - 1, Complex>
routeb_reconstruction_pass3_jet(
    const KerrParameters& parameters, const double radius, const double theta,
    const RouteBReconstructionStateJet<Degree>& state,
    const RouteBReconstructionStateJet<Degree>& sharp_state,
    const Kokkos::Array<RouteBRadialTaylorJet<Degree - 1, Complex>, 4>&
        angular) {
  const auto background =
      routeb_reconstruction_background_jets<Degree - 1>(parameters, radius,
                                                         theta);
  RouteBRadialTaylorJet<Degree - 1, Complex> R =
      RouteBRadialTaylorJet<Degree - 1, Complex>::constant(
          Complex(radius, 0.0));
  if constexpr (Degree > 1) R[1] = Complex(1.0, 0.0);
  const auto R2 = R * R;
  const auto mu_bar = routeb_reconstruction_conjugate(background.mu0);
  const auto pi_bar = routeb_reconstruction_conjugate(background.pi0);
  const auto tau_bar = routeb_reconstruction_conjugate(background.tau0);
  const auto u = state.U.template truncate<Degree - 1>();
  const auto c = state.C.template truncate<Degree - 1>();
  const auto pi = state.Pi.template truncate<Degree - 1>();
  const auto h = state.H.template truncate<Degree - 1>();
  const auto pi_sharp =
      routeb_reconstruction_conjugate(sharp_state.Pi)
          .template truncate<Degree - 1>();
  const auto b_sharp = routeb_reconstruction_conjugate(sharp_state.B)
                           .template truncate<Degree - 1>();
  const auto c_sharp = routeb_reconstruction_conjugate(sharp_state.C)
                           .template truncate<Degree - 1>();
  const auto delta_u =
      -R * mu_bar * u - R * background.mu0 * angular[0] -
      R2 * background.mu0 *
          (pi_bar + Complex(2.0, 0.0) * background.tau0) * c -
      Complex(2.0, 0.0) * angular[1] -
      Complex(2.0, 0.0) * R * pi_bar * pi - Complex(2.0, 0.0) * h -
      Complex(2.0, 0.0) * R * background.pi0 * pi_sharp -
      R * background.pi0 * angular[2] +
      R2 * background.pi0 * background.pi0 * b_sharp +
      R * background.mu0 * angular[3] +
      R2 * (Complex(-3.0, 0.0) * background.mu0 * background.pi0 +
            Complex(2.0, 0.0) * mu_bar * background.pi0 -
            Complex(2.0, 0.0) * background.mu0 * tau_bar) *
          c_sharp;
  return routeb_reconstruction_time_derivative_jet(
      state.U, delta_u, 3, radius, parameters.mass,
      parameters.compactification_length);
}

namespace routeb_reconstruction_detail {

using Jet4 = RouteBRadialTaylorJet<4, Complex>;
using Stride3 = std::array<std::size_t, 3>;
using Stride4 = std::array<std::size_t, 4>;
using Stride5 = std::array<std::size_t, 5>;

KOKKOS_INLINE_FUNCTION std::size_t index3(std::size_t a, std::size_t b,
                                          std::size_t c,
                                          const Stride3& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2];
}
KOKKOS_INLINE_FUNCTION std::size_t index4(std::size_t a, std::size_t b,
                                          std::size_t c, std::size_t d,
                                          const Stride4& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3];
}
KOKKOS_INLINE_FUNCTION std::size_t index5(std::size_t a, std::size_t b,
                                          std::size_t c, std::size_t d,
                                          std::size_t e,
                                          const Stride5& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3] +
         e * stride[4];
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION Jet4 pad(
    const RouteBRadialTaylorJet<Degree, Complex>& input) {
  Jet4 output;
  for (std::size_t order = 0; order <= Degree; ++order) {
    output[order] = input[order];
  }
  return output;
}

template <std::size_t Degree>
KOKKOS_INLINE_FUNCTION RouteBReconstructionStateJet<Degree> load_state(
    const Jet4* jets, const std::size_t level, const std::size_t mode,
    const std::size_t radial, const std::size_t theta,
    const Stride5& stride) {
  RouteBReconstructionStateJet<Degree> state;
  auto load = [&](const ReconstructionField field) {
    return jets[index5(level, mode, static_cast<std::size_t>(field), radial,
                       theta, stride)]
        .template truncate<Degree>();
  };
  state.G = load(ReconstructionField::G);
  state.Lambda = load(ReconstructionField::Lambda);
  state.H = load(ReconstructionField::H);
  state.B = load(ReconstructionField::B);
  state.Pi = load(ReconstructionField::Pi);
  state.C = load(ReconstructionField::C);
  state.U = load(ReconstructionField::U);
  return state;
}

struct ValidateInitial {
  const Complex* input;
  const std::uint64_t* stamps;
  const int* modes;
  const std::size_t* sharp;
  const Real* theta_coordinates;
  std::uint8_t* ready;
  Stride4 input_stride;
  Stride3 stamp_stride;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t sharp_mode = sharp[mode];
    const int signed_mode = modes[mode];
    bool unique = signed_mode != std::numeric_limits<int>::min();
    for (std::size_t other = 0; other < mode_count; ++other) {
      unique = unique && (other == mode || modes[other] != signed_mode);
    }
    bool valid = generation != 0 && unique && sharp_mode < mode_count &&
                 modes[sharp_mode] == -signed_mode &&
                 sharp[sharp_mode] == mode &&
                 stamps[index3(mode, radial, theta, stamp_stride)] ==
                     generation;
    const Real angle = theta_coordinates[theta];
    valid = valid && Kokkos::isfinite(angle) && angle >= 0.0 &&
            angle <= Kokkos::acos(-1.0);
    for (std::size_t field = 0; field < routeb_reconstruction_field_count;
         ++field) {
      const Complex value =
          input[index4(mode, field, radial, theta, input_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag());
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct Initialize {
  const Complex* input;
  const Complex* derivatives;
  Jet4* jets;
  Stride4 input_stride;
  Stride5 derivative_stride;
  Stride5 jet_stride;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    for (std::size_t field = 0; field < routeb_reconstruction_field_count;
         ++field) {
      Jet4 jet;
      jet[0] = input[index4(mode, field, radial, theta, input_stride)];
      for (std::size_t order = 1; order <= 4; ++order) {
        jet[order] =
            derivatives[index5(mode, order - 1, field, radial, theta,
                               derivative_stride)] /
            routeb_factorial(order);
      }
      jets[index5(0, mode, field, radial, theta, jet_stride)] = jet;
    }
  }
};

struct ValidateExternalJet {
  const Complex* values;
  const std::uint64_t* stamps;
  const std::uint64_t* current_level_stamps;
  std::uint8_t* ready;
  Stride4 value_stride;
  Stride4 stamp_stride;
  Stride4 level_stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t active_orders;
  std::uint64_t level_generation;
  std::uint64_t value_stamp;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = current_level_stamps[index4(level, mode, radial, theta,
                                              level_stamp_stride)] ==
                 level_generation;
    for (std::size_t order = 0; order < active_orders; ++order) {
      const Complex value =
          values[index4(mode, order, radial, theta, value_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag()) &&
              stamps[index4(mode, order, radial, theta, stamp_stride)] ==
                  value_stamp;
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct ValidatePass3Angular {
  const Complex* values;
  const std::uint64_t* stamps;
  const std::uint64_t* current_level_stamps;
  std::uint8_t* ready;
  Stride5 value_stride;
  Stride5 stamp_stride;
  Stride4 level_stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t active_orders;
  std::uint64_t level_generation;
  std::uint64_t value_stamp;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = current_level_stamps[index4(level, mode, radial, theta,
                                              level_stamp_stride)] ==
                 level_generation;
    for (std::size_t slot = 0; slot < 4; ++slot) {
      for (std::size_t order = 0; order < active_orders; ++order) {
        const Complex value = values[index5(mode, slot, order, radial, theta,
                                            value_stride)];
        valid = valid && Kokkos::isfinite(value.real()) &&
                Kokkos::isfinite(value.imag()) &&
                stamps[index5(mode, slot, order, radial, theta,
                              stamp_stride)] == value_stamp;
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct ValidateIntermediate {
  const std::uint64_t* coefficient_stamps;
  std::uint8_t* ready;
  Stride5 stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = first_field; field <= last_field; ++field) {
      for (std::size_t order = 0; order < active_orders; ++order) {
        valid = valid &&
                coefficient_stamps[index5(mode, field, order, radial, theta,
                                           stride)] == generation;
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct PassFunctor {
  const Complex* external;
  const int* modes;
  const std::size_t* sharp;
  const Real* theta_coordinates;
  Jet4* jets;
  Stride5 external_stride;
  Stride5 jet_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  int pass;
  UniformRadialGrid grid;
  KerrParameters parameters;

  template <std::size_t Degree>
  KOKKOS_INLINE_FUNCTION void apply(const std::size_t mode,
                                    const std::size_t radial,
                                    const std::size_t theta) const {
    const auto state =
        load_state<Degree>(jets, level, mode, radial, theta, jet_stride);
    const double radius = grid.coordinate(radial);
    if (pass == 1) {
      RouteBRadialTaylorJet<Degree - 1, Complex> psi4;
      RouteBRadialTaylorJet<Degree - 1, Complex> eth1_f;
      for (std::size_t order = 0; order < Degree; ++order) {
        psi4[order] = external[index5(mode, 0, order, radial, theta,
                                      external_stride)];
        eth1_f[order] = external[index5(mode, 1, order, radial, theta,
                                        external_stride)];
      }
      const auto next = routeb_reconstruction_pass1_jet(
          parameters, radius, theta_coordinates[theta], state, psi4, eth1_f);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::G), radial,
                  theta, jet_stride)] = pad(next.G);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::Lambda),
                  radial, theta, jet_stride)] = pad(next.Lambda);
    } else if (pass == 2) {
      RouteBRadialTaylorJet<Degree - 1, Complex> eth2_g;
      for (std::size_t order = 0; order < Degree; ++order) {
        eth2_g[order] = external[index5(mode, 0, order, radial, theta,
                                        external_stride)];
      }
      const auto next = routeb_reconstruction_pass2_jet(
          parameters, radius, theta_coordinates[theta], state, eth2_g);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::H), radial,
                  theta, jet_stride)] = pad(next.H);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::B), radial,
                  theta, jet_stride)] = pad(next.B);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::Pi), radial,
                  theta, jet_stride)] = pad(next.Pi);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::C), radial,
                  theta, jet_stride)] = pad(next.C);
    } else {
      Kokkos::Array<RouteBRadialTaylorJet<Degree - 1, Complex>, 4> angular;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        for (std::size_t order = 0; order < Degree; ++order) {
          angular[slot][order] = external[index5(mode, slot, order, radial,
                                                  theta, external_stride)];
        }
      }
      const auto sharp_state = load_state<Degree>(
          jets, level, sharp[mode], radial, theta, jet_stride);
      const auto next_u = routeb_reconstruction_pass3_jet(
          parameters, radius, theta_coordinates[theta], state, sharp_state,
          angular);
      jets[index5(level + 1, mode,
                  static_cast<std::size_t>(ReconstructionField::U), radial,
                  theta, jet_stride)] = pad(next_u);
    }
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

struct ValidateFields {
  const Jet4* jets;
  std::uint8_t* ready;
  Stride5 stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = first_field; field <= last_field; ++field) {
      const Jet4 jet = jets[index5(level, mode, field, radial, theta, stride)];
      for (std::size_t order = 0; order < active_orders; ++order) {
        valid = valid && Kokkos::isfinite(jet[order].real()) &&
                Kokkos::isfinite(jet[order].imag());
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct PublishFields {
  Jet4* jets;
  Complex* coefficients;
  std::uint64_t* stamps;
  const std::uint8_t* ready;
  Stride5 jet_stride;
  Stride5 coefficient_stride;
  Stride5 stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;
  std::uint64_t generation;
  bool clear_all_on_failure;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    const std::size_t begin = valid || !clear_all_on_failure ? first_field : 0;
    const std::size_t end = valid || !clear_all_on_failure
                                ? last_field
                                : routeb_reconstruction_field_count - 1;
    for (std::size_t field = begin; field <= end; ++field) {
      const Jet4 jet = jets[index5(level, mode, field, radial, theta,
                                   jet_stride)];
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && field >= first_field &&
                            field <= last_field && order < active_orders;
        coefficients[index5(mode, field, order, radial, theta,
                            coefficient_stride)] = active ? jet[order] : Complex{};
        stamps[index5(mode, field, order, radial, theta, stamp_stride)] =
            active ? generation : 0;
      }
      if (!valid) {
        jets[index5(level, mode, field, radial, theta, jet_stride)] = {};
      }
    }
  }
};

struct FinalizeLevel {
  Jet4* jets;
  Complex* values;
  std::uint64_t* level_stamps;
  Complex* current_coefficients;
  std::uint64_t* current_stamps;
  Complex* next_coefficients;
  std::uint64_t* next_stamps;
  const std::uint8_t* ready;
  Stride5 jet_stride;
  Stride5 value_stride;
  Stride4 level_stamp_stride;
  Stride5 coefficient_stride;
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
    for (std::size_t field = 0; field < routeb_reconstruction_field_count;
         ++field) {
      Jet4 jet = jets[index5(level, mode, field, radial, theta, jet_stride)];
      if (!valid) {
        jet = {};
        jets[index5(level, mode, field, radial, theta, jet_stride)] = {};
      }
      values[index5(level, mode, field, radial, theta, value_stride)] =
          valid ? jet[0] : Complex{};
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && order < active_orders;
        current_coefficients[index5(mode, field, order, radial, theta,
                                    coefficient_stride)] =
            active ? jet[order] : Complex{};
        current_stamps[index5(mode, field, order, radial, theta,
                              coefficient_stride)] =
            active ? generation : 0;
        next_coefficients[index5(mode, field, order, radial, theta,
                                 coefficient_stride)] = {};
        next_stamps[index5(mode, field, order, radial, theta,
                           coefficient_stride)] = 0;
      }
    }
    level_stamps[index4(level, mode, radial, theta, level_stamp_stride)] =
        valid ? generation : 0;
  }
};

struct ValidateProjectedRange {
  const Complex* projected;
  const std::uint64_t* stamps;
  std::uint8_t* ready;
  Stride5 projected_stride;
  Stride5 stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;
  std::uint64_t token;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = first_field; field <= last_field; ++field) {
      for (std::size_t order = 0; order < active_orders; ++order) {
        const Complex value = projected[index5(mode, field, order, radial,
                                                theta, projected_stride)];
        valid = valid && Kokkos::isfinite(value.real()) &&
                Kokkos::isfinite(value.imag()) &&
                stamps[index5(mode, field, order, radial, theta,
                              stamp_stride)] == token;
      }
    }
    if (!valid) Kokkos::atomic_exchange(ready, std::uint8_t{0});
  }
};

struct AcceptProjectedNextRange {
  const Complex* projected;
  Jet4* jets;
  Complex* next_coefficients;
  std::uint64_t* next_stamps;
  const std::uint8_t* ready;
  Stride5 projected_stride;
  Stride5 coefficient_stride;
  Stride5 jet_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;
  std::uint64_t token;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    const std::size_t begin = valid ? first_field : 0;
    const std::size_t end = valid ? last_field
                                  : routeb_reconstruction_field_count - 1;
    for (std::size_t field = begin; field <= end; ++field) {
      Jet4 jet = jets[index5(level + 1, mode, field, radial, theta,
                             jet_stride)];
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && order < active_orders;
        const Complex value =
            active ? projected[index5(mode, field, order, radial, theta,
                                      projected_stride)]
                   : Complex{};
        next_coefficients[index5(mode, field, order, radial, theta,
                                 coefficient_stride)] = value;
        next_stamps[index5(mode, field, order, radial, theta,
                           coefficient_stride)] =
            active ? token : 0;
        jet[order] = value;
      }
      jets[index5(level + 1, mode, field, radial, theta, jet_stride)] = jet;
    }
  }
};

struct AcceptProjectedCurrentRange {
  const Complex* projected;
  Jet4* jets;
  Complex* values;
  std::uint64_t* level_stamps;
  Complex* current_coefficients;
  std::uint64_t* current_stamps;
  const std::uint8_t* ready;
  Stride5 projected_stride;
  Stride5 coefficient_stride;
  Stride5 jet_stride;
  Stride5 value_stride;
  Stride4 level_stamp_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t level;
  std::size_t first_field;
  std::size_t last_field;
  std::size_t active_orders;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < routeb_reconstruction_field_count;
         ++field) {
      Jet4 jet = jets[index5(level, mode, field, radial, theta, jet_stride)];
      for (std::size_t order = 0; order < 5; ++order) {
        const bool active = valid && order < active_orders;
        if (!valid || (field >= first_field && field <= last_field)) {
          jet[order] = active
                           ? projected[index5(mode, field, order, radial,
                                              theta, projected_stride)]
                           : Complex{};
        }
        current_coefficients[index5(mode, field, order, radial, theta,
                                    coefficient_stride)] =
            active ? jet[order] : Complex{};
        current_stamps[index5(mode, field, order, radial, theta,
                              coefficient_stride)] =
            active ? generation : 0;
      }
      jets[index5(level, mode, field, radial, theta, jet_stride)] = jet;
      values[index5(level, mode, field, radial, theta, value_stride)] =
          valid ? jet[0] : Complex{};
    }
    level_stamps[index4(level, mode, radial, theta, level_stamp_stride)] =
        valid ? generation : 0;
  }
};

}  // namespace routeb_reconstruction_detail

template <class ExecutionSpace = teuk::ExecutionSpace>
class RouteBReconstructionJetTower {
 public:
  // Queue-order contract: initialize, every external coefficient-wise
  // angular/projection launch, and pass1/pass2/pass3 for a generation must use
  // the same ordered execution-space instance. Methods intentionally do not
  // fence and cannot establish cross-queue dependencies.
  // Pass tokens bind reconstruction angular inputs to a level/pass. Primary
  // psi4/F coefficients carry only the overall generation stamp; selecting
  // coefficients from the matching primary level is an external coordinator
  // contract and cannot be diagnosed by this radial slice.
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using jet_type = routeb_reconstruction_detail::Jet4;
  using value_view =
      Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>;
  using stamp_view =
      Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, memory_space>;
  using coefficient_view =
      Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>;
  using coefficient_stamp_view =
      Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, memory_space>;

  RouteBReconstructionJetTower(
      const std::size_t mode_count, const UniformRadialGrid& grid,
      const std::size_t theta_count,
      const std::string& label = "routeb_reconstruction_jet")
      : mode_count_(mode_count),
        grid_(grid),
        theta_count_(theta_count),
        values_(label + "_values", 5, mode_count, 7, grid.size(), theta_count),
        level_stamps_(label + "_level_stamps", 5, mode_count, grid.size(),
                      theta_count),
        jets_(label + "_jets", 5, mode_count, 7, grid.size(), theta_count),
        input_derivatives_(label + "_input_derivatives", mode_count, 4, 7,
                           grid.size(), theta_count),
        current_coefficients_(label + "_current_coefficients", mode_count, 7,
                              5, grid.size(), theta_count),
        current_coefficient_stamps_(label + "_current_coefficient_stamps",
                                    mode_count, 7, 5, grid.size(), theta_count),
        next_coefficients_(label + "_next_coefficients", mode_count, 7, 5,
                           grid.size(), theta_count),
        next_coefficient_stamps_(label + "_next_coefficient_stamps",
                                 mode_count, 7, 5, grid.size(), theta_count),
        modes_(label + "_modes", mode_count),
        sharp_(label + "_sharp", mode_count),
        theta_(label + "_theta", theta_count),
        ready_(label + "_ready", 1),
        combined_pass1_(label + "_combined_pass1", mode_count, 2, 4,
                        grid.size(), theta_count) {
    if (mode_count == 0 || theta_count == 0 ||
        grid.size() < routeb_fornberg_window) {
      throw std::invalid_argument("Route-B reconstruction extents invalid");
    }
  }

  RouteBReconstructionJetTower(const RouteBReconstructionJetTower&) = delete;
  RouteBReconstructionJetTower& operator=(
      const RouteBReconstructionJetTower&) = delete;
  RouteBReconstructionJetTower(RouteBReconstructionJetTower&&) = delete;
  RouteBReconstructionJetTower& operator=(RouteBReconstructionJetTower&&) =
      delete;

  [[nodiscard]] typename value_view::const_type values() const {
    return values_;
  }
  [[nodiscard]] typename stamp_view::const_type stamps() const {
    return level_stamps_;
  }
  [[nodiscard]] typename coefficient_view::const_type current_coefficients()
      const {
    return current_coefficients_;
  }
  [[nodiscard]] typename coefficient_stamp_view::const_type
  current_coefficient_stamps() const {
    return current_coefficient_stamps_;
  }
  [[nodiscard]] typename coefficient_view::const_type next_coefficients()
      const {
    return next_coefficients_;
  }
  [[nodiscard]] typename coefficient_stamp_view::const_type
  next_coefficient_stamps() const {
    return next_coefficient_stamps_;
  }
  [[nodiscard]] std::size_t current_level() const { return current_level_; }
  [[nodiscard]] std::uint64_t generation() const { return generation_; }
  [[nodiscard]] static bool generation_supported(
      const std::uint64_t generation) {
    return generation != 0 && valid_generation_tokens(generation);
  }
  [[nodiscard]] std::uint64_t expected_pass_token() const {
    if (!initialized_ || phase_ == Phase::Complete) return 0;
    return make_pass_token(current_level_, phase_);
  }
  [[nodiscard]] std::uint64_t expected_initial_projection_token() const {
    return initialized_ && current_level_ == 0 && !initial_projected_
               ? initial_projection_token()
               : 0;
  }

  template <class ModeView, class SharpView, class ThetaView, class InputView,
            class InputStampView>
  void initialize(const execution_space& execution,
                  const KerrParameters& parameters,
                  const ModeView& signed_modes,
                  const SharpView& sharp_indices, const ThetaView& theta,
                  const InputView& input, const InputStampView& input_stamps,
                  const std::uint64_t generation,
                  const ReductionEvolution reduction,
                  const double dissipation_strength) {
    validate_parameters(parameters, reduction, dissipation_strength);
    if (!generation_supported(generation)) {
      throw std::invalid_argument("Route-B reconstruction generation is zero");
    }
    validate_initial_views(signed_modes, sharp_indices, theta, input,
                           input_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    Kokkos::deep_copy(execution, values_, Complex{});
    Kokkos::deep_copy(execution, level_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, jets_, jet_type{});
    Kokkos::deep_copy(execution, current_coefficients_, Complex{});
    Kokkos::deep_copy(execution, current_coefficient_stamps_,
                      std::uint64_t{0});
    Kokkos::deep_copy(execution, next_coefficients_, Complex{});
    Kokkos::deep_copy(execution, next_coefficient_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, modes_, signed_modes);
    Kokkos::deep_copy(execution, sharp_, sharp_indices);
    Kokkos::deep_copy(execution, theta_, theta);
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateInitial{
            input.data(), input_stamps.data(), modes_.data(), sharp_.data(),
            theta_.data(), ready_.data(), strides4(input),
            strides3(input_stamps), mode_count_, grid_.size(), theta_count_,
            generation});
    evaluate_routeb_fornberg_derivatives(execution, input,
                                          input_derivatives_, grid_.spacing());
    Kokkos::parallel_for(
        "routeb_reconstruction_initialize_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::Initialize{
            input.data(), input_derivatives_.data(), jets_.data(),
            strides4(input), strides5(input_derivatives_), strides5(jets_),
            grid_.size(), theta_count_});
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_h0_jets",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateFields{
            jets_.data(), ready_.data(), strides5(jets_), grid_.size(),
            theta_count_, 0, 0, 6, 5});
    Kokkos::parallel_for(
        "routeb_reconstruction_finalize_h0",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::FinalizeLevel{
            jets_.data(), values_.data(), level_stamps_.data(),
            current_coefficients_.data(), current_coefficient_stamps_.data(),
            next_coefficients_.data(), next_coefficient_stamps_.data(),
            ready_.data(), strides5(jets_), strides5(values_),
            strides4(level_stamps_), strides5(current_coefficients_),
            grid_.size(), theta_count_, 0, 5, generation});
    parameters_ = parameters;
    generation_ = generation;
    current_level_ = 0;
    phase_ = Phase::Pass1;
    initial_projected_ = false;
    pass1_projected_ = false;
    pass2_projected_ = false;
    pass3_projected_ = false;
    initialized_ = true;
  }

  template <class PsiView, class PsiStampView, class AngularView,
            class AngularStampView>
  void pass1(const execution_space& execution, const PsiView& psi4,
             const PsiStampView& psi4_stamps, const AngularView& eth1_f,
             const AngularStampView& eth1_f_stamps,
             const std::uint64_t generation,
             const std::uint64_t pass_token,
             const ReductionEvolution reduction,
             const double dissipation_strength) {
    require_phase(Phase::Pass1, generation, pass_token, reduction,
                  dissipation_strength);
    validate_jet_views(psi4, psi4_stamps);
    validate_jet_views(eth1_f, eth1_f_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    Kokkos::deep_copy(execution, next_coefficients_, Complex{});
    Kokkos::deep_copy(execution, next_coefficient_stamps_, std::uint64_t{0});
    const std::size_t active = 4 - current_level_;
    validate_external(execution, psi4, psi4_stamps, active, generation_,
                      "routeb_reconstruction_validate_psi4");
    validate_external(execution, eth1_f, eth1_f_stamps, active, pass_token,
                      "routeb_reconstruction_validate_eth1f");
    pack_pass1(execution, psi4, eth1_f);
    launch_pass(execution, combined_pass1_, 1, 0, 1, active, pass_token);
    phase_ = Phase::Pass2;
    pass1_projected_ = false;
  }

  template <class AngularView, class AngularStampView>
  void pass2(const execution_space& execution, const AngularView& eth2_g,
             const AngularStampView& eth2_g_stamps,
             const std::uint64_t generation,
             const std::uint64_t pass_token,
             const ReductionEvolution reduction,
             const double dissipation_strength) {
    require_phase(Phase::Pass2, generation, pass_token, reduction,
                  dissipation_strength);
    validate_jet_views(eth2_g, eth2_g_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t active = 4 - current_level_;
    validate_intermediate(execution, 0, 1, active,
                          make_pass_token(current_level_, Phase::Pass1));
    validate_external(execution, eth2_g, eth2_g_stamps, active, pass_token,
                      "routeb_reconstruction_validate_eth2g");
    pack_single(execution, eth2_g);
    launch_pass(execution, combined_pass1_, 2, 2, 5, active, pass_token);
    phase_ = Phase::Pass3;
    pass2_projected_ = false;
  }

  template <class AngularView, class AngularStampView>
  void pass3(const execution_space& execution, const AngularView& angular,
             const AngularStampView& angular_stamps,
             const std::uint64_t generation,
             const std::uint64_t pass_token,
             const ReductionEvolution reduction,
             const double dissipation_strength) {
    require_phase(Phase::Pass3, generation, pass_token, reduction,
                  dissipation_strength);
    validate_pass3_views(angular, angular_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t active = 4 - current_level_;
    validate_intermediate(execution, 0, 1, active,
                          make_pass_token(current_level_, Phase::Pass1));
    validate_intermediate(execution, 2, 5, active,
                          make_pass_token(current_level_, Phase::Pass2));
    validate_pass3_external(execution, angular, angular_stamps, active,
                            pass_token);
    launch_pass(execution, angular, 3, 6, 6, active, pass_token);
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_finalize_level",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::FinalizeLevel{
            jets_.data(), values_.data(), level_stamps_.data(),
            current_coefficients_.data(), current_coefficient_stamps_.data(),
            next_coefficients_.data(), next_coefficient_stamps_.data(),
            ready_.data(), strides5(jets_), strides5(values_),
            strides4(level_stamps_), strides5(current_coefficients_),
            grid_.size(), theta_count_, current_level_ + 1, active,
            generation_});
    ++current_level_;
    phase_ = current_level_ < 4 ? Phase::Pass1 : Phase::Complete;
    pass3_projected_ = false;
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_initial_projection(
      const execution_space& execution, const ProjectedView& projected,
      const ProjectedStampView& projected_stamps,
      const std::uint64_t generation, const std::uint64_t pass_token) {
    if (!initialized_ || current_level_ != 0 || phase_ != Phase::Pass1 ||
        generation != generation_ ||
        initial_projected_ || pass_token != initial_projection_token()) {
      throw std::logic_error(
          "Route-B reconstruction h0 projection is unavailable");
    }
    validate_projected_views(projected, projected_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_h0_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateProjectedRange{
            projected.data(), projected_stamps.data(), ready_.data(),
            strides5(projected), strides5(projected_stamps), grid_.size(),
            theta_count_, 0, routeb_reconstruction_field_count - 1, 5,
            pass_token});
    Kokkos::parallel_for(
        "routeb_reconstruction_accept_h0_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::AcceptProjectedCurrentRange{
            projected.data(), jets_.data(), values_.data(),
            level_stamps_.data(), current_coefficients_.data(),
            current_coefficient_stamps_.data(), ready_.data(),
            strides5(projected), strides5(current_coefficients_),
            strides5(jets_), strides5(values_), strides4(level_stamps_),
            grid_.size(), theta_count_, 0, 0,
            routeb_reconstruction_field_count - 1, 5, generation_});
    initial_projected_ = true;
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_pass1_projection(
      const execution_space& execution, const ProjectedView& projected,
      const ProjectedStampView& projected_stamps,
      const std::uint64_t generation, const std::uint64_t pass_token) {
    if (!initialized_ || phase_ != Phase::Pass2 || pass1_projected_ ||
        generation != generation_ ||
        pass_token != make_pass_token(current_level_, Phase::Pass1)) {
      throw std::logic_error(
          "Route-B reconstruction pass1 projection is unavailable");
    }
    accept_projected_next(execution, projected, projected_stamps, 0, 1,
                          pass_token);
    pass1_projected_ = true;
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_pass2_projection(
      const execution_space& execution, const ProjectedView& projected,
      const ProjectedStampView& projected_stamps,
      const std::uint64_t generation, const std::uint64_t pass_token) {
    if (!initialized_ || phase_ != Phase::Pass3 || pass2_projected_ ||
        generation != generation_ ||
        pass_token != make_pass_token(current_level_, Phase::Pass2)) {
      throw std::logic_error(
          "Route-B reconstruction pass2 projection is unavailable");
    }
    accept_projected_next(execution, projected, projected_stamps, 2, 5,
                          pass_token);
    pass2_projected_ = true;
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_pass3_projection(
      const execution_space& execution, const ProjectedView& projected,
      const ProjectedStampView& projected_stamps,
      const std::uint64_t generation, const std::uint64_t pass_token) {
    if (!initialized_ || current_level_ == 0 || pass3_projected_ ||
        generation != generation_ ||
        (phase_ != Phase::Pass1 && phase_ != Phase::Complete) ||
        pass_token != make_pass_token(current_level_ - 1, Phase::Pass3)) {
      throw std::logic_error(
          "Route-B reconstruction pass3 projection is unavailable");
    }
    validate_projected_views(projected, projected_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t active = 5 - current_level_;
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_pass3_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateProjectedRange{
            projected.data(), projected_stamps.data(), ready_.data(),
            strides5(projected), strides5(projected_stamps), grid_.size(),
            theta_count_, 6, 6, active, pass_token});
    Kokkos::parallel_for(
        "routeb_reconstruction_accept_pass3_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::AcceptProjectedCurrentRange{
            projected.data(), jets_.data(), values_.data(),
            level_stamps_.data(), current_coefficients_.data(),
            current_coefficient_stamps_.data(), ready_.data(),
            strides5(projected), strides5(current_coefficients_),
            strides5(jets_), strides5(values_), strides4(level_stamps_),
            grid_.size(), theta_count_, current_level_, 6, 6, active,
            generation_});
    pass3_projected_ = true;
  }

 private:
  enum class Phase { Pass1, Pass2, Pass3, Complete };

  [[nodiscard]] std::uint64_t initial_projection_token() const {
    return generation_ ^ 0x6eed0e9da4d94a4fULL;
  }

  [[nodiscard]] std::uint64_t make_pass_token(const std::size_t level,
                                              const Phase phase) const {
    return make_pass_token_for_generation(generation_, level, phase);
  }

  [[nodiscard]] static std::uint64_t make_pass_token_for_generation(
      const std::uint64_t generation, const std::size_t level,
      const Phase phase) {
    const std::uint64_t phase_code =
        phase == Phase::Pass1 ? 1 : phase == Phase::Pass2 ? 2 : 3;
    return generation ^ (0x9e3779b97f4a7c15ULL * (level + 1)) ^
           (0xd1b54a32d192ed03ULL * phase_code);
  }

  [[nodiscard]] static bool valid_generation_tokens(
      const std::uint64_t generation) {
    std::array<std::uint64_t, 13> tokens{};
    std::size_t index = 1;
    tokens[0] = generation ^ 0x6eed0e9da4d94a4fULL;
    if (tokens[0] == 0) return false;
    for (std::size_t level = 0; level < 4; ++level) {
      for (const Phase phase : {Phase::Pass1, Phase::Pass2, Phase::Pass3}) {
        const auto token =
            make_pass_token_for_generation(generation, level, phase);
        if (token == 0) return false;
        for (std::size_t previous = 0; previous < index; ++previous) {
          if (tokens[previous] == token) return false;
        }
        tokens[index++] = token;
      }
    }
    return true;
  }

  template <class View>
  static routeb_reconstruction_detail::Stride3 strides3(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2)};
  }
  template <class View>
  static routeb_reconstruction_detail::Stride4 strides4(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2), view.stride(3)};
  }
  template <class View>
  static routeb_reconstruction_detail::Stride5 strides5(const View& view) {
    return {view.stride(0), view.stride(1), view.stride(2), view.stride(3),
            view.stride(4)};
  }

  static void validate_parameters(const KerrParameters& parameters,
                                  const ReductionEvolution reduction,
                                  const double dissipation) {
    if (reduction != ReductionEvolution::FreeDamped || dissipation != 0.0 ||
        !(parameters.mass > 0.0) || !std::isfinite(parameters.mass) ||
        !std::isfinite(parameters.spin) ||
        std::abs(parameters.spin) > parameters.mass ||
        !(parameters.compactification_length > 0.0) ||
        !std::isfinite(parameters.compactification_length)) {
      throw std::invalid_argument("Route-B reconstruction policy invalid");
    }
  }

  template <class View>
  bool overlaps_owned(const View& view) const {
    return routeb_fornberg_detail::allocations_overlap(view, values_) ||
           routeb_fornberg_detail::allocations_overlap(view, level_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view, jets_) ||
           routeb_fornberg_detail::allocations_overlap(view,
                                                        input_derivatives_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, current_coefficients_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, current_coefficient_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view,
                                                        next_coefficients_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, next_coefficient_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view, modes_) ||
           routeb_fornberg_detail::allocations_overlap(view, sharp_) ||
           routeb_fornberg_detail::allocations_overlap(view, theta_) ||
           routeb_fornberg_detail::allocations_overlap(view, ready_) ||
           routeb_fornberg_detail::allocations_overlap(view, combined_pass1_);
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
             view.extent(1) == routeb_reconstruction_field_count &&
             view.extent(2) == 5 && view.extent(3) == grid_.size() &&
             view.extent(4) == theta_count_ &&
             routeb_fornberg_detail::has_separated_strides<5>(view) &&
             !overlaps_owned(view);
    };
    if (!valid(projected) || !valid(projected_stamps) ||
        routeb_fornberg_detail::allocations_overlap(projected,
                                                     projected_stamps)) {
      throw std::invalid_argument(
          "Route-B reconstruction projected coefficient views invalid");
    }
  }

  template <class ProjectedView, class ProjectedStampView>
  void accept_projected_next(const execution_space& execution,
                             const ProjectedView& projected,
                             const ProjectedStampView& projected_stamps,
                             const std::size_t first_field,
                             const std::size_t last_field,
                             const std::uint64_t pass_token) {
    validate_projected_views(projected, projected_stamps);
    Kokkos::deep_copy(execution, ready_, std::uint8_t{1});
    const std::size_t active = 4 - current_level_;
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_next_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateProjectedRange{
            projected.data(), projected_stamps.data(), ready_.data(),
            strides5(projected), strides5(projected_stamps), grid_.size(),
            theta_count_, first_field, last_field, active, pass_token});
    Kokkos::parallel_for(
        "routeb_reconstruction_accept_next_projection",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::AcceptProjectedNextRange{
            projected.data(), jets_.data(), next_coefficients_.data(),
            next_coefficient_stamps_.data(), ready_.data(),
            strides5(projected), strides5(next_coefficients_),
            strides5(jets_), grid_.size(), theta_count_, current_level_,
            first_field, last_field, active, pass_token});
  }

  template <class ModeView, class SharpView, class ThetaView, class InputView,
            class StampView>
  void validate_initial_views(const ModeView& modes, const SharpView& sharp,
                              const ThetaView& theta, const InputView& input,
                              const StampView& stamps) const {
    static_assert(ModeView::rank == 1 && SharpView::rank == 1 &&
                  ThetaView::rank == 1 && InputView::rank == 4 &&
                  StampView::rank == 3);
    static_assert(
        std::is_same_v<typename ModeView::non_const_value_type, int> &&
        std::is_same_v<typename SharpView::non_const_value_type, std::size_t> &&
        std::is_same_v<typename ThetaView::non_const_value_type, Real> &&
        std::is_same_v<typename InputView::non_const_value_type, Complex> &&
        std::is_same_v<typename StampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(
        Kokkos::SpaceAccessibility<execution_space,
                                   typename ModeView::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename SharpView::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename ThetaView::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename InputView::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename StampView::memory_space>::accessible);
    if (modes.data() == nullptr || sharp.data() == nullptr ||
        theta.data() == nullptr || input.data() == nullptr ||
        stamps.data() == nullptr || modes.extent(0) != mode_count_ ||
        sharp.extent(0) != mode_count_ || theta.extent(0) != theta_count_ ||
        input.extent(0) != mode_count_ || input.extent(1) != 7 ||
        input.extent(2) != grid_.size() ||
        input.extent(3) != theta_count_ || stamps.extent(0) != mode_count_ ||
        stamps.extent(1) != grid_.size() ||
        stamps.extent(2) != theta_count_ ||
        !routeb_fornberg_detail::has_separated_strides<1>(modes) ||
        !routeb_fornberg_detail::has_separated_strides<1>(sharp) ||
        !routeb_fornberg_detail::has_separated_strides<1>(theta) ||
        !routeb_fornberg_detail::has_separated_strides<4>(input) ||
        !routeb_fornberg_detail::has_separated_strides<3>(stamps) ||
        overlaps_owned(modes) || overlaps_owned(sharp) ||
        overlaps_owned(theta) || overlaps_owned(input) ||
        overlaps_owned(stamps)) {
      throw std::invalid_argument("Route-B reconstruction h0 views invalid");
    }
  }

  template <class View, class StampView>
  void validate_jet_views(const View& view, const StampView& stamps) const {
    static_assert(View::rank == 4 && StampView::rank == 4);
    static_assert(
        std::is_same_v<typename View::non_const_value_type, Complex> &&
        std::is_same_v<typename StampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(
        Kokkos::SpaceAccessibility<execution_space,
                                   typename View::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename StampView::memory_space>::accessible);
    if (view.data() == nullptr || stamps.data() == nullptr ||
        view.extent(0) != mode_count_ || view.extent(1) != 4 ||
        view.extent(2) != grid_.size() || view.extent(3) != theta_count_ ||
        stamps.extent(0) != mode_count_ || stamps.extent(1) != 4 ||
        stamps.extent(2) != grid_.size() ||
        stamps.extent(3) != theta_count_ ||
        !routeb_fornberg_detail::has_separated_strides<4>(view) ||
        !routeb_fornberg_detail::has_separated_strides<4>(stamps) ||
        overlaps_owned(view) || overlaps_owned(stamps)) {
      throw std::invalid_argument("Route-B reconstruction jet views invalid");
    }
  }

  template <class View, class StampView>
  void validate_pass3_views(const View& view, const StampView& stamps) const {
    static_assert(View::rank == 5 && StampView::rank == 5);
    static_assert(
        std::is_same_v<typename View::non_const_value_type, Complex> &&
        std::is_same_v<typename StampView::non_const_value_type,
                       std::uint64_t>);
    static_assert(
        Kokkos::SpaceAccessibility<execution_space,
                                   typename View::memory_space>::accessible &&
        Kokkos::SpaceAccessibility<execution_space,
                                   typename StampView::memory_space>::accessible);
    if (view.data() == nullptr || stamps.data() == nullptr ||
        view.extent(0) != mode_count_ || view.extent(1) != 4 ||
        view.extent(2) != 4 || view.extent(3) != grid_.size() ||
        view.extent(4) != theta_count_ || stamps.extent(0) != mode_count_ ||
        stamps.extent(1) != 4 || stamps.extent(2) != 4 ||
        stamps.extent(3) != grid_.size() ||
        stamps.extent(4) != theta_count_ ||
        !routeb_fornberg_detail::has_separated_strides<5>(view) ||
        !routeb_fornberg_detail::has_separated_strides<5>(stamps) ||
        overlaps_owned(view) || overlaps_owned(stamps)) {
      throw std::invalid_argument(
          "Route-B reconstruction pass3 views invalid");
    }
  }

  void require_phase(const Phase expected, const std::uint64_t generation,
                     const std::uint64_t pass_token,
                     const ReductionEvolution reduction,
                     const double dissipation) const {
    if (!initialized_ || phase_ != expected || generation != generation_ ||
        pass_token == 0 || pass_token != make_pass_token(current_level_, expected)) {
      throw std::logic_error("Route-B reconstruction phase unavailable");
    }
    if (reduction != ReductionEvolution::FreeDamped || dissipation != 0.0) {
      throw std::invalid_argument(
          "Route-B reconstruction requires FreeDamped zero dissipation");
    }
  }

  template <class View, class StampView>
  void validate_external(const execution_space& execution, const View& view,
                         const StampView& stamps, const std::size_t active,
                         const std::uint64_t expected_stamp,
                         const char* label) {
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        label, Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateExternalJet{
            view.data(), stamps.data(), level_stamps_.data(), ready_.data(),
            strides4(view), strides4(stamps), strides4(level_stamps_),
            grid_.size(), theta_count_, current_level_, active,
            generation_, expected_stamp});
  }

  template <class PsiView, class AngularView>
  void pack_pass1(const execution_space& execution, const PsiView& psi4,
                  const AngularView& eth1_f) {
    const std::size_t radial_count = grid_.size();
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count_ * 4 * radial_count * theta_count;
    const auto combined = combined_pass1_;
    Kokkos::parallel_for(
        "routeb_reconstruction_pack_pass1",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = 4 * radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within_mode = flat - mode * plane;
          const std::size_t order = within_mode / (radial_count * theta_count);
          const std::size_t within =
              within_mode - order * radial_count * theta_count;
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within - radial * theta_count;
          combined(mode, 0, order, radial, theta) =
              psi4(mode, order, radial, theta);
          combined(mode, 1, order, radial, theta) =
              eth1_f(mode, order, radial, theta);
        });
  }

  template <class View>
  void pack_single(const execution_space& execution, const View& input) {
    const std::size_t radial_count = grid_.size();
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count_ * 4 * radial_count * theta_count;
    const auto combined = combined_pass1_;
    Kokkos::parallel_for(
        "routeb_reconstruction_pack_single",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = 4 * radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within_mode = flat - mode * plane;
          const std::size_t order = within_mode / (radial_count * theta_count);
          const std::size_t within =
              within_mode - order * radial_count * theta_count;
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within - radial * theta_count;
          combined(mode, 0, order, radial, theta) =
              input(mode, order, radial, theta);
        });
  }

  template <class ExternalView>
  void launch_pass(const execution_space& execution,
                   const ExternalView& external, const int pass,
                   const std::size_t first_field,
                   const std::size_t last_field,
                   const std::size_t active,
                   const std::uint64_t published_stamp) {
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        pass == 1 ? "routeb_reconstruction_pass1"
                  : pass == 2 ? "routeb_reconstruction_pass2"
                              : "routeb_reconstruction_pass3",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::PassFunctor{
            external.data(), modes_.data(), sharp_.data(), theta_.data(),
            jets_.data(), strides5(external), strides5(jets_), grid_.size(),
            theta_count_, current_level_, pass, grid_, parameters_});
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_pass_fields",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateFields{
            jets_.data(), ready_.data(), strides5(jets_), grid_.size(),
            theta_count_, current_level_ + 1, first_field, last_field, active});
    Kokkos::parallel_for(
        "routeb_reconstruction_publish_pass_fields",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::PublishFields{
            jets_.data(), next_coefficients_.data(),
            next_coefficient_stamps_.data(), ready_.data(), strides5(jets_),
            strides5(next_coefficients_), strides5(next_coefficient_stamps_),
            grid_.size(), theta_count_, current_level_ + 1, first_field,
            last_field, active, published_stamp, true});
  }

  void validate_intermediate(const execution_space& execution,
                             const std::size_t first_field,
                             const std::size_t last_field,
                             const std::size_t active,
                             const std::uint64_t expected_stamp) {
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_intermediate",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidateIntermediate{
            next_coefficient_stamps_.data(), ready_.data(),
            strides5(next_coefficient_stamps_), grid_.size(), theta_count_,
            first_field, last_field, active, expected_stamp});
  }

  template <class View, class StampView>
  void validate_pass3_external(const execution_space& execution,
                               const View& view, const StampView& stamps,
                               const std::size_t active,
                               const std::uint64_t expected_stamp) {
    const std::size_t total = mode_count_ * grid_.size() * theta_count_;
    Kokkos::parallel_for(
        "routeb_reconstruction_validate_pass3_angular",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_reconstruction_detail::ValidatePass3Angular{
            view.data(), stamps.data(), level_stamps_.data(), ready_.data(),
            strides5(view), strides5(stamps), strides4(level_stamps_),
            grid_.size(), theta_count_, current_level_, active,
            generation_, expected_stamp});
  }

  std::size_t mode_count_;
  UniformRadialGrid grid_;
  std::size_t theta_count_;
  value_view values_;
  stamp_view level_stamps_;
  Kokkos::View<jet_type*****, Kokkos::LayoutRight, memory_space> jets_;
  Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>
      input_derivatives_;
  coefficient_view current_coefficients_;
  coefficient_stamp_view current_coefficient_stamps_;
  coefficient_view next_coefficients_;
  coefficient_stamp_view next_coefficient_stamps_;
  Kokkos::View<int*, memory_space> modes_;
  Kokkos::View<std::size_t*, memory_space> sharp_;
  Kokkos::View<Real*, memory_space> theta_;
  Kokkos::View<std::uint8_t*, memory_space> ready_;
  Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space> combined_pass1_;
  KerrParameters parameters_;
  std::uint64_t generation_ = 0;
  std::size_t current_level_ = 0;
  Phase phase_ = Phase::Complete;
  bool initialized_ = false;
  bool initial_projected_ = false;
  bool pass1_projected_ = false;
  bool pass2_projected_ = false;
  bool pass3_projected_ = false;
};

}  // namespace teuk
