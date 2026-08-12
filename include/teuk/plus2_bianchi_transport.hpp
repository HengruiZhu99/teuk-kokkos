#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "teuk/angular_coordinator.hpp"
#include "teuk/background.hpp"
#include "teuk/device_rk4.hpp"
#include "teuk/ghp.hpp"
#include "teuk/grid.hpp"
#include "teuk/jet.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_bianchi.hpp"
#include "teuk/plus2_transported_curvature.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class Plus2BianchiStateComponent : std::size_t {
  Z0 = 0,
  Z1 = 1,
  Count = 2,
};

// Complete h[0..2] and stationary-background inputs used by (B0), (B1), and
// their exact first time derivatives.  Bianchi-5 couples Sig to the stationary
// type-D curvature psi20, so its tangent contains SigT but no H/HT factor.
// Delta3Psi20 and EthPrime3Psi20 are explicit because this layer does not
// introduce an unreviewed background derivative formula.
enum class Plus2BianchiPrimaryComponent : std::size_t {
  H = 0,
  HT = 1,
  HTT = 2,
  CSharp = 3,
  CSharpT = 4,
  BSharp = 5,
  BSharpT = 6,
  Ta = 7,
  TaT = 8,
  Sig = 9,
  SigT = 10,
  Delta3Psi20 = 11,
  EthPrime3Psi20 = 12,
  Count = 13,
};

enum class Plus2BianchiConstraintComponent : std::size_t {
  Delta4Z1 = 0,
  Delta5Z0 = 1,
  Delta4Z1T = 2,
  Delta5Z0T = 3,
  Count = 4,
};

using Plus2BianchiStateView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiConstStateView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiPrimaryView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiPrimaryStampView =
    Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiConstraintView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2BianchiThetaView = Kokkos::View<Real*, MemorySpace>;

struct Plus2BianchiInitializationContract {
  // A durable evidence identifier is mandatory so an initialization cannot
  // become "qualified" through a default-constructed boolean bundle.
  std::string evidence_id;
  bool metric_curvature_constraints_qualified = false;
  bool radial_boundary_data_qualified = false;
  bool independently_qualified_scri_coefficients = false;
  std::string boundary_evidence_id;
};

struct Plus2BianchiStageCapability {
  std::string boundary_evidence_id;
  bool common_rk_stage = false;
  bool complete_h_through_second_tangent = false;
  bool angular_band_qualified = false;
  bool radial_boundary_treatment_qualified = false;
  bool independently_qualified_scri_coefficients = false;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
};

struct Plus2BianchiPrimaryWriteTarget {
  std::uint64_t generation = 0;
  Plus2BianchiPrimaryView fields;
  Plus2BianchiPrimaryStampView stamps;
};

namespace plus2_bianchi_transport_detail {

enum class ScratchComponent : std::size_t {
  DrZ0 = 0,
  DrZ1 = 1,
  DrZ0T = 2,
  DrZ1T = 3,
  Eth3H = 4,
  Eth3HT = 5,
  Eth4Z1 = 6,
  Eth4Z1T = 7,
  EthPrime4Z1 = 8,
  EthPrime4Z1T = 9,
  Eth5Z0 = 10,
  Eth5Z0T = 11,
  F0 = 12,
  F1 = 13,
  F0T = 14,
  F1T = 15,
  Count = 16,
};

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

struct SetScalarReadyFunctor {
  std::uint8_t* ready;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t) const {
    ready[0] = static_cast<std::uint8_t>(1);
  }
};

struct ValidateInputsFunctor {
  const Complex* state;
  const Complex* primary;
  const std::uint64_t* stamps;
  std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t state_count =
        static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
    constexpr std::size_t primary_count =
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = generation != 0;
    for (std::size_t field = 0; field < state_count; ++field) {
      const Complex value = state[flat4(mode, field, radial, theta,
                                        state_count, radial_count,
                                        theta_count)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag());
    }
    for (std::size_t field = 0; field < primary_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      primary_count, radial_count,
                                      theta_count);
      const Complex value = primary[index];
      valid = valid && stamps[index] == generation &&
              Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag());
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct StateRadialDerivativeFunctor {
  const Complex* state;
  Complex* scratch;
  RadialDiscretization discretization;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t state_count =
        static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % state_count;
    const std::size_t mode = mode_field / state_count;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t input = flat4(mode, field, 0, theta, state_count,
                                    radial_count, theta_count);
    const std::size_t output_field =
        field == static_cast<std::size_t>(Plus2BianchiStateComponent::Z0)
            ? static_cast<std::size_t>(ScratchComponent::DrZ0)
            : static_cast<std::size_t>(ScratchComponent::DrZ1);
    scratch[flat4(mode, output_field, radial, theta, scratch_count,
                  radial_count, theta_count)] =
        radial_first_derivative_strided_at(
            discretization, state + input, radial_count, radial,
            inverse_spacing, theta_count);
  }
};

struct TangentRadialDerivativeFunctor {
  const Complex* curvature;
  Complex* scratch;
  RadialDiscretization discretization;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    constexpr std::size_t tangent_count = 2;
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % tangent_count;
    const std::size_t mode = mode_field / tangent_count;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t curvature_field =
        field == 0
            ? static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z0T)
            : static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z1T);
    const std::size_t output_field =
        field == 0 ? static_cast<std::size_t>(ScratchComponent::DrZ0T)
                   : static_cast<std::size_t>(ScratchComponent::DrZ1T);
    const std::size_t input =
        flat4(mode, curvature_field, 0, theta, curvature_count,
              radial_count, theta_count);
    scratch[flat4(mode, output_field, radial, theta, scratch_count,
                  radial_count, theta_count)] =
        radial_first_derivative_strided_at(
            discretization, curvature + input, radial_count, radial,
            inverse_spacing, theta_count);
  }
};

struct ComputeZ1TFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* state;
  const Complex* primary;
  Complex* curvature;
  Complex* scratch;
  Complex* rhs;
  const std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t state_count =
        static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
    constexpr std::size_t primary_count =
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto state_at = [&](const Plus2BianchiStateComponent component) {
      return state[flat4(mode, static_cast<std::size_t>(component), radial,
                         theta, state_count, radial_count, theta_count)];
    };
    const auto primary_at =
        [&](const Plus2BianchiPrimaryComponent component) {
          return primary[flat4(mode, static_cast<std::size_t>(component),
                               radial, theta, primary_count, radial_count,
                               theta_count)];
        };
    const auto scratch_at = [&](const ScratchComponent component) {
      return scratch[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    const Complex z0 = state_at(Plus2BianchiStateComponent::Z0);
    const Complex z1 = state_at(Plus2BianchiStateComponent::Z1);
    Complex f1{};
    Complex z1t{};
    if (ready[0]) {
      const double radius = grid.coordinate(radial);
      const auto background = kerr_background_point(
          parameters, radius, cos_theta[theta], sin_theta[theta]);
      f1 = plus2_bianchi_delta4_z1(
          radius, background, z1,
          primary_at(Plus2BianchiPrimaryComponent::H),
          primary_at(Plus2BianchiPrimaryComponent::CSharp),
          primary_at(Plus2BianchiPrimaryComponent::BSharp),
          primary_at(Plus2BianchiPrimaryComponent::Ta),
          scratch_at(ScratchComponent::Eth3H),
          primary_at(Plus2BianchiPrimaryComponent::Delta3Psi20),
          primary_at(Plus2BianchiPrimaryComponent::EthPrime3Psi20));
      z1t = plus2_invert_capital_delta_n(
          f1, z1, scratch_at(ScratchComponent::DrZ1), 4, radius,
          parameters.mass, parameters.compactification_length);
    }
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z0),
        radial, theta, curvature_count, radial_count, theta_count)] =
        ready[0] ? z0 : Complex{};
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z1),
        radial, theta, curvature_count, radial_count, theta_count)] =
        ready[0] ? z1 : Complex{};
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z1T),
        radial, theta, curvature_count, radial_count, theta_count)] = z1t;
    scratch[flat4(mode, static_cast<std::size_t>(ScratchComponent::F1),
                  radial, theta, scratch_count, radial_count,
                  theta_count)] = f1;
    rhs[flat4(mode,
              static_cast<std::size_t>(Plus2BianchiStateComponent::Z1),
              radial, theta, state_count, radial_count, theta_count)] = z1t;
  }
};

struct ComputeZ0TFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* primary;
  Complex* curvature;
  Complex* scratch;
  Complex* rhs;
  const std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t state_count =
        static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
    constexpr std::size_t primary_count =
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto curvature_at =
        [&](const Plus2TransportedCurvatureComponent component) {
          return curvature[flat4(mode, static_cast<std::size_t>(component),
                                 radial, theta, curvature_count, radial_count,
                                 theta_count)];
        };
    const auto primary_at =
        [&](const Plus2BianchiPrimaryComponent component) {
          return primary[flat4(mode, static_cast<std::size_t>(component),
                               radial, theta, primary_count, radial_count,
                               theta_count)];
        };
    const auto scratch_at = [&](const ScratchComponent component) {
      return scratch[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    Complex f0{};
    Complex z0t{};
    if (ready[0]) {
      const double radius = grid.coordinate(radial);
      const auto background = kerr_background_point(
          parameters, radius, cos_theta[theta], sin_theta[theta]);
      f0 = plus2_bianchi_delta5_z0(
          radius, background,
          curvature_at(Plus2TransportedCurvatureComponent::Z0),
          curvature_at(Plus2TransportedCurvatureComponent::Z1),
          primary_at(Plus2BianchiPrimaryComponent::Sig),
          scratch_at(ScratchComponent::Eth4Z1));
      z0t = plus2_invert_capital_delta_n(
          f0, curvature_at(Plus2TransportedCurvatureComponent::Z0),
          scratch_at(ScratchComponent::DrZ0), 5, radius, parameters.mass,
          parameters.compactification_length);
    }
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z0T),
        radial, theta, curvature_count, radial_count, theta_count)] = z0t;
    scratch[flat4(mode, static_cast<std::size_t>(ScratchComponent::F0),
                  radial, theta, scratch_count, radial_count,
                  theta_count)] = f0;
    rhs[flat4(mode,
              static_cast<std::size_t>(Plus2BianchiStateComponent::Z0),
              radial, theta, state_count, radial_count, theta_count)] = z0t;
  }
};

struct ComputeSecondTangentsFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* primary;
  Complex* curvature;
  Complex* scratch;
  const std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t primary_count =
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto c = [&](const Plus2TransportedCurvatureComponent component) {
      return curvature[flat4(mode, static_cast<std::size_t>(component), radial,
                             theta, curvature_count, radial_count,
                             theta_count)];
    };
    const auto p = [&](const Plus2BianchiPrimaryComponent component) {
      return primary[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, primary_count, radial_count, theta_count)];
    };
    const auto s = [&](const ScratchComponent component) {
      return scratch[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    Complex f1t{};
    Complex z1tt{};
    if (ready[0]) {
      const double radius = grid.coordinate(radial);
      const auto background = kerr_background_point(
          parameters, radius, cos_theta[theta], sin_theta[theta]);
      const auto f1 = plus2_bianchi_delta4_z1(
          radius, background,
          Jet1<Complex>{c(Plus2TransportedCurvatureComponent::Z1),
                        c(Plus2TransportedCurvatureComponent::Z1T)},
          Jet1<Complex>{p(Plus2BianchiPrimaryComponent::H),
                        p(Plus2BianchiPrimaryComponent::HT)},
          Jet1<Complex>{p(Plus2BianchiPrimaryComponent::CSharp),
                        p(Plus2BianchiPrimaryComponent::CSharpT)},
          Jet1<Complex>{p(Plus2BianchiPrimaryComponent::BSharp),
                        p(Plus2BianchiPrimaryComponent::BSharpT)},
          Jet1<Complex>{p(Plus2BianchiPrimaryComponent::Ta),
                        p(Plus2BianchiPrimaryComponent::TaT)},
          Jet1<Complex>{s(ScratchComponent::Eth3H),
                        s(ScratchComponent::Eth3HT)},
          p(Plus2BianchiPrimaryComponent::Delta3Psi20),
          p(Plus2BianchiPrimaryComponent::EthPrime3Psi20));
      f1t = f1.dt;
      z1tt = plus2_invert_capital_delta_n(
          f1t, c(Plus2TransportedCurvatureComponent::Z1T),
          s(ScratchComponent::DrZ1T), 4, radius, parameters.mass,
          parameters.compactification_length);
    }
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z1TT),
        radial, theta, curvature_count, radial_count, theta_count)] = z1tt;
    scratch[flat4(mode, static_cast<std::size_t>(ScratchComponent::F1T),
                  radial, theta, scratch_count, radial_count,
                  theta_count)] = f1t;
  }
};

struct ComputeZ0TTAndConstraintsFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* primary;
  Complex* curvature;
  Complex* scratch;
  Complex* constraints;
  const std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t primary_count =
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    constexpr std::size_t constraint_count =
        static_cast<std::size_t>(Plus2BianchiConstraintComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto c = [&](const Plus2TransportedCurvatureComponent component) {
      return curvature[flat4(mode, static_cast<std::size_t>(component), radial,
                             theta, curvature_count, radial_count,
                             theta_count)];
    };
    const auto p = [&](const Plus2BianchiPrimaryComponent component) {
      return primary[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, primary_count, radial_count, theta_count)];
    };
    const auto s = [&](const ScratchComponent component) {
      return scratch[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    Complex f0t{};
    Complex z0tt{};
    Complex residuals[constraint_count]{};
    if (ready[0]) {
      const double radius = grid.coordinate(radial);
      const auto background = kerr_background_point(
          parameters, radius, cos_theta[theta], sin_theta[theta]);
      const auto f0 = plus2_bianchi_delta5_z0(
          radius, background,
          Jet1<Complex>{c(Plus2TransportedCurvatureComponent::Z0),
                        c(Plus2TransportedCurvatureComponent::Z0T)},
          Jet1<Complex>{c(Plus2TransportedCurvatureComponent::Z1),
                        c(Plus2TransportedCurvatureComponent::Z1T)},
          Jet1<Complex>{p(Plus2BianchiPrimaryComponent::Sig),
                        p(Plus2BianchiPrimaryComponent::SigT)},
          Jet1<Complex>{s(ScratchComponent::Eth4Z1),
                        s(ScratchComponent::Eth4Z1T)});
      f0t = f0.dt;
      z0tt = plus2_invert_capital_delta_n(
          f0t, c(Plus2TransportedCurvatureComponent::Z0T),
          s(ScratchComponent::DrZ0T), 5, radius, parameters.mass,
          parameters.compactification_length);
      residuals[0] =
          delta_n_point(c(Plus2TransportedCurvatureComponent::Z1),
                        c(Plus2TransportedCurvatureComponent::Z1T),
                        s(ScratchComponent::DrZ1), 4, radius,
                        parameters.mass, parameters.compactification_length) -
          s(ScratchComponent::F1);
      residuals[1] =
          delta_n_point(c(Plus2TransportedCurvatureComponent::Z0),
                        c(Plus2TransportedCurvatureComponent::Z0T),
                        s(ScratchComponent::DrZ0), 5, radius,
                        parameters.mass, parameters.compactification_length) -
          s(ScratchComponent::F0);
      residuals[2] = delta_n_point(
                         c(Plus2TransportedCurvatureComponent::Z1T),
                         c(Plus2TransportedCurvatureComponent::Z1TT),
                         s(ScratchComponent::DrZ1T), 4, radius,
                         parameters.mass,
                         parameters.compactification_length) -
                     s(ScratchComponent::F1T);
      residuals[3] = delta_n_point(
                         c(Plus2TransportedCurvatureComponent::Z0T), z0tt,
                         s(ScratchComponent::DrZ0T), 5, radius,
                         parameters.mass,
                         parameters.compactification_length) -
                     f0t;
    }
    curvature[flat4(
        mode, static_cast<std::size_t>(
                  Plus2TransportedCurvatureComponent::Z0TT),
        radial, theta, curvature_count, radial_count, theta_count)] = z0tt;
    scratch[flat4(mode, static_cast<std::size_t>(ScratchComponent::F0T),
                  radial, theta, scratch_count, radial_count,
                  theta_count)] = f0t;
    for (std::size_t field = 0; field < constraint_count; ++field) {
      constraints[flat4(mode, field, radial, theta, constraint_count,
                        radial_count, theta_count)] = residuals[field];
    }
  }
};

struct FinalizeStageFunctor {
  const Complex* scratch;
  Complex* derivatives;
  std::uint64_t* curvature_stamps;
  std::uint64_t* derivative_stamps;
  const std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(ScratchComponent::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t derivative_count =
        static_cast<std::size_t>(Plus2BianchiDerivativeComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto scratch_at = [&](const ScratchComponent component) {
      return scratch[flat4(mode, static_cast<std::size_t>(component), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    const Complex values[derivative_count]{
        scratch_at(ScratchComponent::F1),
        scratch_at(ScratchComponent::F1T),
        scratch_at(ScratchComponent::EthPrime4Z1),
        scratch_at(ScratchComponent::EthPrime4Z1T),
        scratch_at(ScratchComponent::F0),
        scratch_at(ScratchComponent::F0T),
        scratch_at(ScratchComponent::Eth5Z0),
        scratch_at(ScratchComponent::Eth5Z0T)};
    for (std::size_t field = 0; field < derivative_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      derivative_count, radial_count,
                                      theta_count);
      derivatives[index] = ready[0] ? values[field] : Complex{};
      derivative_stamps[index] = ready[0] ? generation : 0;
    }
    for (std::size_t field = 0; field < curvature_count; ++field) {
      curvature_stamps[flat4(mode, field, radial, theta, curvature_count,
                             radial_count, theta_count)] =
          ready[0] ? generation : 0;
    }
  }
};

static_assert(std::is_trivially_copyable_v<SetScalarReadyFunctor>);
static_assert(std::is_trivially_copyable_v<ValidateInputsFunctor>);
static_assert(std::is_trivially_copyable_v<StateRadialDerivativeFunctor>);
static_assert(std::is_trivially_copyable_v<TangentRadialDerivativeFunctor>);
static_assert(std::is_trivially_copyable_v<ComputeZ1TFunctor>);
static_assert(std::is_trivially_copyable_v<ComputeZ0TFunctor>);
static_assert(std::is_trivially_copyable_v<ComputeSecondTangentsFunctor>);
static_assert(
    std::is_trivially_copyable_v<ComputeZ0TTAndConstraintsFunctor>);
static_assert(std::is_trivially_copyable_v<FinalizeStageFunctor>);
static_assert(sizeof(ComputeZ1TFunctor) < 1800);
static_assert(sizeof(ComputeZ0TTAndConstraintsFunctor) < 1800);

}  // namespace plus2_bianchi_transport_detail

template <class ExecSpace = ExecutionSpace>
class Plus2BianchiTransport {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename execution_space::memory_space;
  static_assert(std::is_same_v<memory_space, MemorySpace>,
                "transport adapter must share the live-source memory space");
  using flat_view = Kokkos::View<
      Complex*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using const_flat_view = Kokkos::View<
      const Complex*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  Plus2BianchiTransport(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& radial_grid, const KerrParameters& parameters,
      const int ell_max, const Plus2BianchiThetaView& cos_theta,
      const Plus2BianchiThetaView& sin_theta,
      const RadialDiscretization radial_discretization =
          RadialDiscretization::D105,
      const std::string& label = "plus2_bianchi_transport")
      : registry_(registry),
        radial_grid_(radial_grid),
        parameters_(parameters),
        ell_max_(ell_max),
        radial_discretization_(radial_discretization),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        radius_(label + "_radius", radial_grid.size()),
        state_(label + "_state", registry.size(),
               static_cast<std::size_t>(Plus2BianchiStateComponent::Count),
               radial_grid.size(), sin_theta.extent(0)),
        primary_(label + "_primary", registry.size(),
                 static_cast<std::size_t>(
                     Plus2BianchiPrimaryComponent::Count),
                 radial_grid.size(), sin_theta.extent(0)),
        primary_stamps_(label + "_primary_stamps", registry.size(),
                        static_cast<std::size_t>(
                            Plus2BianchiPrimaryComponent::Count),
                        radial_grid.size(), sin_theta.extent(0)),
        curvature_(label + "_curvature", registry.size(),
                   static_cast<std::size_t>(
                       Plus2TransportedCurvatureComponent::Count),
                   radial_grid.size(), sin_theta.extent(0)),
        curvature_stamps_(label + "_curvature_stamps", registry.size(),
                          static_cast<std::size_t>(
                              Plus2TransportedCurvatureComponent::Count),
                          radial_grid.size(), sin_theta.extent(0)),
        derivatives_(label + "_derivatives", registry.size(),
                     static_cast<std::size_t>(
                         Plus2BianchiDerivativeComponent::Count),
                     radial_grid.size(), sin_theta.extent(0)),
        derivative_stamps_(label + "_derivative_stamps", registry.size(),
                           static_cast<std::size_t>(
                               Plus2BianchiDerivativeComponent::Count),
                           radial_grid.size(), sin_theta.extent(0)),
        constraints_(label + "_constraints", registry.size(),
                     static_cast<std::size_t>(
                         Plus2BianchiConstraintComponent::Count),
                     radial_grid.size(), sin_theta.extent(0)),
        scratch_(label + "_scratch", registry.size(),
                 static_cast<std::size_t>(
                     plus2_bianchi_transport_detail::ScratchComponent::Count),
                 radial_grid.size(), sin_theta.extent(0)),
        ready_(label + "_ready", 1),
        h_angular_(execution, registry, 0, 0, ell_max,
                   static_cast<int>(sin_theta.extent(0)), radial_grid.size(),
                   parameters),
        z1_angular_(execution, registry, 1, 1, ell_max,
                    static_cast<int>(sin_theta.extent(0)), radial_grid.size(),
                    parameters),
        z0_angular_(execution, registry, 2, 2, ell_max,
                    static_cast<int>(sin_theta.extent(0)), radial_grid.size(),
                    parameters),
        rk_workspace_(state_.size()) {
    if (registry.size() == 0 || !registry.is_closed_under_sharp() ||
        ell_max < 2 || radial_grid.size() <
                           radial_minimum_points(radial_discretization) ||
        cos_theta.extent(0) == 0 ||
        cos_theta.extent(0) != sin_theta.extent(0) ||
        !std::isfinite(parameters.mass) || !std::isfinite(parameters.spin) ||
        !std::isfinite(parameters.compactification_length) ||
        parameters.mass <= 0.0 ||
        std::abs(parameters.spin) > parameters.mass ||
        parameters.compactification_length <= 0.0) {
      throw std::invalid_argument("invalid spin +2 Bianchi transport setup");
    }
    auto radius_host = Kokkos::create_mirror_view(radius_);
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      radius_host(radial) = radial_grid.coordinate(radial);
    }
    Kokkos::deep_copy(execution, radius_, radius_host);
    Kokkos::deep_copy(execution, primary_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, curvature_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, derivative_stamps_, std::uint64_t{0});
  }

  [[nodiscard]] std::size_t mode_count() const noexcept {
    return registry_.size();
  }
  [[nodiscard]] std::size_t radial_count() const noexcept {
    return radial_grid_.size();
  }
  [[nodiscard]] std::size_t theta_count() const noexcept {
    return sin_theta_.extent(0);
  }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }
  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return radial_discretization_;
  }
  [[nodiscard]] const ModeRegistry& registry() const noexcept {
    return registry_;
  }
  [[nodiscard]] const UniformRadialGrid& radial_grid() const noexcept {
    return radial_grid_;
  }
  [[nodiscard]] const KerrParameters& parameters() const noexcept {
    return parameters_;
  }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] const std::string& initialization_evidence_id() const noexcept {
    return initialization_evidence_id_;
  }
  [[nodiscard]] const std::string& boundary_evidence_id() const noexcept {
    return boundary_evidence_id_;
  }
  [[nodiscard]] bool scri_coefficients_qualified() const noexcept {
    return scri_coefficients_qualified_;
  }
  [[nodiscard]] bool boundary_data_qualified() const noexcept {
    return boundary_data_qualified_;
  }
  [[nodiscard]] std::uint64_t last_generation() const noexcept {
    return last_generation_;
  }
  [[nodiscard]] Plus2BianchiConstStateView state() const { return state_; }
  [[nodiscard]] Plus2BianchiConstraintView constraints() const {
    return constraints_;
  }
  [[nodiscard]] Kokkos::View<const std::uint8_t*, MemorySpace> readiness()
      const {
    return ready_;
  }
  [[nodiscard]] Plus2TransportedCurvatureStage latest_stage() const {
    if (!stage_available_) {
      throw std::logic_error("spin +2 Bianchi stage is not available");
    }
    return {curvature_, curvature_stamps_};
  }
  [[nodiscard]] Plus2BianchiDerivativeStage latest_derivative_stage() const {
    if (!stage_available_) {
      throw std::logic_error("spin +2 Bianchi stage is not available");
    }
    return {derivatives_, derivative_stamps_};
  }

  template <class InitialView>
  void initialize(const execution_space& execution,
                  const InitialView& initial_state,
                  const Plus2BianchiInitializationContract& contract) {
    static_assert(InitialView::rank == 4,
                  "Bianchi initial state must be rank four");
    validate_initialization_contract(contract);
    if (initial_state.extent(0) != registry_.size() ||
        initial_state.extent(1) != static_cast<std::size_t>(
                                       Plus2BianchiStateComponent::Count) ||
        initial_state.extent(2) != radial_grid_.size() ||
        initial_state.extent(3) != sin_theta_.extent(0)) {
      throw std::invalid_argument("Bianchi initial state extent mismatch");
    }
    Kokkos::deep_copy(execution, state_, initial_state);
    initialization_evidence_id_ = contract.evidence_id;
    boundary_evidence_id_ = contract.boundary_evidence_id;
    scri_coefficients_qualified_ =
        contract.independently_qualified_scri_coefficients;
    boundary_data_qualified_ = contract.radial_boundary_data_qualified;
    initialized_ = true;
    stage_available_ = false;
    last_generation_ = 0;
    next_generation_ = 1;
  }

  template <class CheckpointView>
  void restore_checkpoint_state(
      const execution_space& execution,
      const CheckpointView& checkpoint_state,
      const Plus2BianchiInitializationContract& contract,
      const std::uint64_t last_generation) {
    if (last_generation == std::numeric_limits<std::uint64_t>::max()) {
      throw std::invalid_argument(
          "spin +2 Bianchi checkpoint generation is exhausted");
    }
    initialize(execution, checkpoint_state, contract);
    last_generation_ = last_generation;
    next_generation_ = last_generation + 1;
  }

  [[nodiscard]] flat_view flat_state() const {
    return flat_view(state_.data(), state_.size());
  }
  [[nodiscard]] DeviceRK4Workspace<Complex, execution_space>& rk_workspace() {
    return rk_workspace_;
  }

  template <class PrimaryStateView, class PrimaryInputProducer>
  Plus2TransportedCurvatureStage evaluate_stage(
      const execution_space& execution, const double stage_time,
      const PrimaryStateView& primary_stage, const const_flat_view& state_stage,
      const flat_view& rhs_output, const std::uint64_t generation,
      const Plus2BianchiStageCapability& capability,
      PrimaryInputProducer&& primary_input_producer) {
    validate_stage_contract(capability, state_stage, rhs_output, generation);
    using primary_view_type = std::remove_cvref_t<PrimaryStateView>;
    using const_primary_view = typename primary_view_type::const_type;
    const const_primary_view read_only_primary = primary_stage;
    primary_input_producer(
        execution, stage_time, read_only_primary,
        Plus2BianchiPrimaryWriteTarget{generation, primary_,
                                       primary_stamps_});
    enqueue_stage(execution, state_stage, rhs_output, generation);
    stage_available_ = true;
    last_generation_ = generation;
    next_generation_ = generation + 1;
    return {curvature_, curvature_stamps_};
  }

  [[nodiscard]] std::uint64_t next_generation() const {
    if (next_generation_ == 0) {
      throw std::overflow_error("spin +2 Bianchi generation exhausted");
    }
    return next_generation_;
  }

 private:
  void validate_initialization_contract(
      const Plus2BianchiInitializationContract& contract) const {
    const bool has_scri = radial_grid_.lower_radius() == 0.0;
    if (contract.evidence_id.empty() ||
        contract.boundary_evidence_id.empty() ||
        !contract.metric_curvature_constraints_qualified ||
        !contract.radial_boundary_data_qualified ||
        (has_scri &&
         !contract.independently_qualified_scri_coefficients)) {
      throw std::invalid_argument(
          "spin +2 Bianchi initialization evidence is incomplete");
    }
  }

  void validate_stage_contract(
      const Plus2BianchiStageCapability& capability,
      const const_flat_view& state_stage, const flat_view& rhs_output,
      const std::uint64_t generation) const {
    const bool has_scri = radial_grid_.lower_radius() == 0.0;
    if (!initialized_ ||
        capability.boundary_evidence_id != boundary_evidence_id_ ||
        !capability.common_rk_stage ||
        !capability.complete_h_through_second_tangent ||
        !capability.angular_band_qualified ||
        !capability.radial_boundary_treatment_qualified ||
        capability.radial_discretization != radial_discretization_ ||
        !boundary_data_qualified_ ||
        (has_scri &&
         (!scri_coefficients_qualified_ ||
          !capability.independently_qualified_scri_coefficients)) ||
        generation == 0 || generation != next_generation_ ||
        state_stage.extent(0) != state_.size() ||
        rhs_output.extent(0) != state_.size() ||
        state_stage.data() == rhs_output.data()) {
      throw std::invalid_argument(
          "spin +2 Bianchi common-stage contract is incomplete");
    }
  }

  void enqueue_stage(const execution_space& execution,
                     const const_flat_view& state_stage,
                     const flat_view& rhs_output,
                     const std::uint64_t generation) {
    using namespace plus2_bianchi_transport_detail;
    const std::size_t modes = registry_.size();
    const std::size_t radial = radial_grid_.size();
    const std::size_t theta = sin_theta_.extent(0);
    const std::size_t point_count = modes * radial * theta;
    const std::size_t state_count =
        static_cast<std::size_t>(Plus2BianchiStateComponent::Count);
    Kokkos::parallel_for(
        "plus2_bianchi_set_ready",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        SetScalarReadyFunctor{ready_.data()});
    Kokkos::parallel_for(
        "plus2_bianchi_validate_stage_inputs",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        ValidateInputsFunctor{state_stage.data(), primary_.data(),
                              primary_stamps_.data(), ready_.data(),
                              generation, radial, theta});
    Kokkos::parallel_for(
        "plus2_bianchi_state_radial_derivatives",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             modes * state_count * radial *
                                                 theta),
        StateRadialDerivativeFunctor{
            state_stage.data(), scratch_.data(), radial_discretization_, modes,
            radial, theta, 1.0 / radial_grid_.spacing()});
    h_angular_.eth(
        execution, primary_,
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::H), primary_,
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::HT), radius_,
        sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth3H));
    h_angular_.eth(
        execution, primary_,
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::HT), primary_,
        static_cast<std::size_t>(Plus2BianchiPrimaryComponent::HTT), radius_,
        sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth3HT));
    Kokkos::parallel_for(
        "plus2_bianchi_compute_z1_t",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        ComputeZ1TFunctor{radial_grid_, parameters_, cos_theta_.data(),
                          sin_theta_.data(), state_stage.data(),
                          primary_.data(), curvature_.data(), scratch_.data(),
                          rhs_output.data(), ready_.data(), radial, theta});
    z1_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth4Z1));
    z1_angular_.ethprime(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::EthPrime4Z1));
    Kokkos::parallel_for(
        "plus2_bianchi_compute_z0_t",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        ComputeZ0TFunctor{radial_grid_, parameters_, cos_theta_.data(),
                          sin_theta_.data(), primary_.data(),
                          curvature_.data(), scratch_.data(), rhs_output.data(),
                          ready_.data(), radial, theta});
    Kokkos::parallel_for(
        "plus2_bianchi_tangent_radial_derivatives",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             modes * 2 * radial * theta),
        TangentRadialDerivativeFunctor{
            curvature_.data(), scratch_.data(), radial_discretization_, radial,
            theta, 1.0 / radial_grid_.spacing()});
    Kokkos::parallel_for(
        "plus2_bianchi_compute_z1_tt",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        ComputeSecondTangentsFunctor{
            radial_grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
            primary_.data(), curvature_.data(), scratch_.data(), ready_.data(),
            radial, theta});
    z1_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1TT),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth4Z1T));
    z1_angular_.ethprime(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1TT),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::EthPrime4Z1T));
    Kokkos::parallel_for(
        "plus2_bianchi_compute_z0_tt_constraints_and_stamps",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        ComputeZ0TTAndConstraintsFunctor{
            radial_grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
            primary_.data(), curvature_.data(), scratch_.data(),
            constraints_.data(), ready_.data(), radial, theta});
    z0_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0T),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth5Z0));
    z0_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0T),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0TT),
        radius_, sin_theta_, cos_theta_, scratch_,
        static_cast<std::size_t>(ScratchComponent::Eth5Z0T));
    Kokkos::parallel_for(
        "plus2_bianchi_finalize_stage_derivatives_and_stamps",
        Kokkos::RangePolicy<execution_space>(execution, 0, point_count),
        FinalizeStageFunctor{scratch_.data(), derivatives_.data(),
                             curvature_stamps_.data(),
                             derivative_stamps_.data(), ready_.data(),
                             generation, radial, theta});
  }

  ModeRegistry registry_;
  UniformRadialGrid radial_grid_;
  KerrParameters parameters_;
  int ell_max_;
  RadialDiscretization radial_discretization_;
  Plus2BianchiThetaView cos_theta_;
  Plus2BianchiThetaView sin_theta_;
  Kokkos::View<Real*, MemorySpace> radius_;
  Plus2BianchiStateView state_;
  Plus2BianchiPrimaryView primary_;
  Plus2BianchiPrimaryStampView primary_stamps_;
  Plus2TransportedCurvatureStorageView curvature_;
  Plus2LiveStampView curvature_stamps_;
  Plus2BianchiDerivativeView derivatives_;
  Plus2LiveStampView derivative_stamps_;
  Plus2BianchiConstraintView constraints_;
  Plus2BianchiPrimaryView scratch_;
  Kokkos::View<std::uint8_t*, MemorySpace> ready_;
  SignedModeAngularCoordinator<execution_space> h_angular_;
  SignedModeAngularCoordinator<execution_space> z1_angular_;
  SignedModeAngularCoordinator<execution_space> z0_angular_;
  DeviceRK4Workspace<Complex, execution_space> rk_workspace_;
  bool initialized_ = false;
  bool stage_available_ = false;
  bool scri_coefficients_qualified_ = false;
  bool boundary_data_qualified_ = false;
  std::string initialization_evidence_id_;
  std::string boundary_evidence_id_;
  std::uint64_t last_generation_ = 0;
  std::uint64_t next_generation_ = 1;
};

// One-way common-stage RK4 wrapper.  primary_rhs cannot access the curvature
// transport through its signature; the passive stage sees the primary stage,
// builds h[0..2], computes Route-A closure/tangents, and then exposes the
// generation-stamped curvature and derivative stages to the observer
// (normally the live source gate).
template <class PrimaryValue, class ExecutionSpace, class PrimaryStateView,
          class PrimaryRightHandSide, class PrimaryInputProducer,
          class StageObserver>
void device_one_way_bianchi_transport_rk4_step(
    const ExecutionSpace& execution, const PrimaryStateView& primary,
    const double time, const double step, PrimaryRightHandSide&& primary_rhs,
    PrimaryInputProducer&& primary_input_producer,
    StageObserver&& stage_observer,
    DeviceRK4Workspace<PrimaryValue, ExecutionSpace>& primary_workspace,
    Plus2BianchiTransport<ExecutionSpace>& transport,
    const Plus2BianchiStageCapability& capability) {
  if (!transport.initialized()) {
    throw std::logic_error("spin +2 Bianchi transport is not initialized");
  }
  auto companion_rhs =
      [&](const ExecutionSpace& stage_execution, const double stage_time,
          const auto& primary_stage, const auto& curvature_stage,
          const auto& output) {
        using transport_type = Plus2BianchiTransport<ExecutionSpace>;
        const typename transport_type::const_flat_view state(
            curvature_stage.data(), curvature_stage.extent(0));
        const typename transport_type::flat_view rhs(output.data(),
                                                      output.extent(0));
        const std::uint64_t generation = transport.next_generation();
        const auto adapter = transport.evaluate_stage(
            stage_execution, stage_time, primary_stage, state, rhs,
            generation, capability, primary_input_producer);
        stage_observer(stage_execution, stage_time, generation, adapter,
                       transport.latest_derivative_stage());
      };
  device_one_way_coupled_rk4_step(
      execution, primary, transport.flat_state(), time, step,
      std::forward<PrimaryRightHandSide>(primary_rhs), companion_rhs,
      primary_workspace, transport.rk_workspace());
}

// Standalone three-state common-stage validation coordinator.  The companion
// callback is invoked only after the passive Bianchi RHS has exposed the
// curvature and derivative adapters for that exact primary/Bianchi RK stage.
// The rotating Route-A transport is weakly hyperbolic and this wrapper must
// not be used as a production runtime path.  Its callback signature is
//
//   companion_rhs(exec, time, generation, primary_stage, bianchi_stage,
//                 curvature, derivatives, companion_stage, companion_out)
//
// The primary RHS has no Bianchi or companion argument, and the Bianchi RHS
// has no companion argument, so the one-way dependency is structural.  This
// wrapper allocates no storage and performs no fence.
template <class PrimaryValue, class CompanionValue, class ExecutionSpace,
          class PrimaryStateView, class CompanionStateView,
          class PrimaryRightHandSide, class PrimaryInputProducer,
          class CompanionRightHandSide>
void device_one_way_bianchi_companion_rk4_step(
    const ExecutionSpace& execution, const PrimaryStateView& primary,
    const CompanionStateView& companion, const double time, const double step,
    PrimaryRightHandSide&& primary_rhs,
    PrimaryInputProducer&& primary_input_producer,
    CompanionRightHandSide&& companion_rhs,
    DeviceRK4Workspace<PrimaryValue, ExecutionSpace>& primary_workspace,
    Plus2BianchiTransport<ExecutionSpace>& transport,
    DeviceRK4Workspace<CompanionValue, ExecutionSpace>& companion_workspace,
    const Plus2BianchiStageCapability& capability) {
  if (!transport.initialized()) {
    throw std::logic_error("spin +2 Bianchi transport is not initialized");
  }
  std::uint64_t stage_generation = 0;
  Plus2TransportedCurvatureStage curvature;
  Plus2BianchiDerivativeStage derivatives;
  auto bianchi_rhs =
      [&](const ExecutionSpace& stage_execution, const double stage_time,
          const auto& primary_stage, const auto& bianchi_stage,
          const auto& output) {
        using transport_type = Plus2BianchiTransport<ExecutionSpace>;
        const typename transport_type::const_flat_view state(
            bianchi_stage.data(), bianchi_stage.extent(0));
        const typename transport_type::flat_view rhs(output.data(),
                                                      output.extent(0));
        stage_generation = transport.next_generation();
        curvature = transport.evaluate_stage(
            stage_execution, stage_time, primary_stage, state, rhs,
            stage_generation, capability, primary_input_producer);
        derivatives = transport.latest_derivative_stage();
      };
  auto passive_rhs =
      [&](const ExecutionSpace& stage_execution, const double stage_time,
          const auto& primary_stage, const auto& bianchi_stage,
          const auto& companion_stage, const auto& output) {
        using transport_type = Plus2BianchiTransport<ExecutionSpace>;
        const typename transport_type::const_flat_view read_only_bianchi(
            bianchi_stage.data(), bianchi_stage.extent(0));
        companion_rhs(stage_execution, stage_time, stage_generation,
                      primary_stage, read_only_bianchi, curvature, derivatives,
                      companion_stage, output);
      };
  device_one_way_three_state_rk4_step(
      execution, primary, transport.flat_state(), companion, time, step,
      std::forward<PrimaryRightHandSide>(primary_rhs), bianchi_rhs,
      passive_rhs, primary_workspace, transport.rk_workspace(),
      companion_workspace);
}

}  // namespace teuk
