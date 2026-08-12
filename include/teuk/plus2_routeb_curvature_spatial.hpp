#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/angular_coordinator.hpp"
#include "teuk/plus2_endpoint_extraction.hpp"
#include "teuk/plus2_source_primitive_spatial.hpp"
#include "teuk/plus2_transported_curvature.hpp"

namespace teuk {

inline constexpr std::uint32_t plus2_routeb_curvature_provider_version = 1;
inline constexpr const char* plus2_routeb_curvature_provider_name =
    "routeb-five-level-constrained-positive-node-v1";

using Plus2RouteBTowerView =
    Kokkos::View<Complex*****, Kokkos::LayoutRight, MemorySpace>;
using Plus2RouteBConstTowerView =
    Kokkos::View<const Complex*****, Kokkos::LayoutRight, MemorySpace>;
using Plus2RouteBTowerStampView =
    Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2RouteBConstTowerStampView =
    Kokkos::View<const std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;

// Complete same-stage h[0]..h[4] reconstruction tower.  The level stamp is
// pointwise because the reviewed Route-B tower finalizes all seven
// reconstruction fields atomically at each (level,mode,R,theta) point.
struct Plus2RouteBCurvatureTowerStage {
  std::uint64_t generation = 0;
  Plus2RouteBConstTowerView fields;
  Plus2RouteBConstTowerStampView stamps;
};

struct Plus2RouteBCurvatureOffsets {
  std::size_t H = 2;
  std::size_t B = 3;
  std::size_t Pi = 4;
  std::size_t C = 5;
  std::size_t U = 6;
};

namespace plus2_routeb_curvature_detail {

using PrimitiveScratch = plus2_primitive_spatial_detail::Scratch;

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat5(
    const std::size_t level, const std::size_t mode,
    const std::size_t field, const std::size_t radial,
    const std::size_t theta, const std::size_t mode_count,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((((level * mode_count + mode) * field_count + field) *
            radial_count +
            radial) *
           theta_count) +
         theta;
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t stamp4(
    const std::size_t level, const std::size_t mode,
    const std::size_t radial, const std::size_t theta,
    const std::size_t mode_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return (((level * mode_count + mode) * radial_count + radial) *
          theta_count) +
         theta;
}

enum class Profile : std::size_t {
  F0_0 = 0,
  F1_0 = 1,
  Z0Regular_0 = 2,
  Z1Regular_0 = 3,
  F0_1 = 4,
  F1_1 = 5,
  Z0Regular_1 = 6,
  Z1Regular_1 = 7,
  F0_2 = 8,
  F1_2 = 9,
  Z0Regular_2 = 10,
  Z1Regular_2 = 11,
  Count = 12,
};

enum class CurvatureRadial : std::size_t {
  DrZ0 = 0,
  DrZ1 = 1,
  DrZ0T = 2,
  DrZ1T = 3,
  Count = 4,
};

struct SetReadyFunctor {
  std::uint8_t* ready;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t) const {
    ready[0] = static_cast<std::uint8_t>(1);
  }
};

struct ValidateTowerFunctor {
  const Complex* fields;
  const std::uint64_t* stamps;
  std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t mode_count;
  std::size_t field_count;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t H;
  std::size_t B;
  std::size_t Pi;
  std::size_t C;
  std::size_t U;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t selected[5]{H, B, Pi, C, U};
    bool valid = generation != 0;
    for (std::size_t level = 0; level < 5; ++level) {
      valid = valid &&
              stamps[stamp4(level, mode, radial, theta, mode_count,
                            radial_count, theta_count)] == generation;
      for (const std::size_t field : selected) {
        const Complex value = fields[flat5(
            level, mode, field, radial, theta, mode_count, field_count,
            radial_count, theta_count)];
        valid = valid && Kokkos::isfinite(value.real()) &&
                Kokkos::isfinite(value.imag());
      }
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

// Route B is qualified on the fixed 9/17/33 direct-Fornberg windows.  Do not
// route these derivatives through the D10-5 SBP operator: doing so would both
// change the reviewed radial graph and make the required nine-point window
// unavailable.
struct MetricRadialFunctor {
  const Complex* scratch_input;
  Complex* scratch_output;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(PrimitiveScratch::Count);
    constexpr PrimitiveScratch inputs[10]{
        PrimitiveScratch::BSharp,  PrimitiveScratch::BSharpT,
        PrimitiveScratch::CSharp,  PrimitiveScratch::CSharpT,
        PrimitiveScratch::C,       PrimitiveScratch::CT,
        PrimitiveScratch::V,       PrimitiveScratch::VT,
        PrimitiveScratch::VSharp,  PrimitiveScratch::VSharpT};
    constexpr PrimitiveScratch outputs[10]{
        PrimitiveScratch::DrBSharp,  PrimitiveScratch::DrBSharpT,
        PrimitiveScratch::DrCSharp,  PrimitiveScratch::DrCSharpT,
        PrimitiveScratch::DrC,       PrimitiveScratch::DrCT,
        PrimitiveScratch::DrV,       PrimitiveScratch::DrVT,
        PrimitiveScratch::DrVSharp,  PrimitiveScratch::DrVSharpT};
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % 10;
    const std::size_t mode = mode_field / 10;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t input = flat4(
        mode, static_cast<std::size_t>(inputs[field]), 0, theta, count,
        radial_count, theta_count);
    scratch_output[flat4(mode, static_cast<std::size_t>(outputs[field]),
                         radial, theta, count, radial_count, theta_count)] =
        routeb_fornberg_direct_derivative_at(
            1, scratch_input + input, radial_count, radial, inverse_spacing,
            theta_count);
  }
};

struct ConnectionRadialFunctor {
  const Complex* scratch_input;
  Complex* scratch_output;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(PrimitiveScratch::Count);
    constexpr PrimitiveScratch inputs[3]{PrimitiveScratch::Sig,
                                         PrimitiveScratch::Kap,
                                         PrimitiveScratch::Be};
    constexpr PrimitiveScratch outputs[3]{PrimitiveScratch::DrSig,
                                          PrimitiveScratch::DrKap,
                                          PrimitiveScratch::DrBe};
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % 3;
    const std::size_t mode = mode_field / 3;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t input = flat4(
        mode, static_cast<std::size_t>(inputs[field]), 0, theta, count,
        radial_count, theta_count);
    scratch_output[flat4(mode, static_cast<std::size_t>(outputs[field]),
                         radial, theta, count, radial_count, theta_count)] =
        routeb_fornberg_direct_derivative_at(
            1, scratch_input + input, radial_count, radial, inverse_spacing,
            theta_count);
  }
};

struct CurvatureProfileFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const int* modes;
  const Complex* scratch;
  Complex* profiles;
  std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t time_level;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t scratch_count =
        static_cast<std::size_t>(PrimitiveScratch::Count);
    constexpr std::size_t profile_count =
        static_cast<std::size_t>(Profile::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto s = [&](const PrimitiveScratch field) {
      return scratch[flat4(mode, static_cast<std::size_t>(field), radial,
                           theta, scratch_count, radial_count, theta_count)];
    };
    const double radius = grid.coordinate(radial);
    const auto background = plus2_primitive_background(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    const Complex thorn_sig = thorn_n_point(
        s(PrimitiveScratch::Sig), s(PrimitiveScratch::SigT),
        s(PrimitiveScratch::DrSig), 2, 2, 1, modes[mode], radius,
        cos_theta[theta], parameters.mass, parameters.spin,
        parameters.compactification_length, background.kerr.epsilon0);
    const Complex thorn_be = thorn_n_point(
        s(PrimitiveScratch::Be), s(PrimitiveScratch::BeT),
        s(PrimitiveScratch::DrBe), 2, 1, 0, modes[mode], radius,
        cos_theta[theta], parameters.mass, parameters.spin,
        parameters.compactification_length, background.kerr.epsilon0);
    const Plus2ReconstructionPrimitiveInputs metric{
        s(PrimitiveScratch::U),
        s(PrimitiveScratch::USharp),
        s(PrimitiveScratch::C),
        s(PrimitiveScratch::CSharp),
        s(PrimitiveScratch::B),
        s(PrimitiveScratch::BSharp),
        s(PrimitiveScratch::H),
        s(PrimitiveScratch::Pi)};
    Plus2PrimitiveDerivatives derivatives{};
    derivatives.thorn1_Bsharp = s(PrimitiveScratch::Thorn1BSharp);
    derivatives.thorn2_Csharp = s(PrimitiveScratch::Thorn2CSharp);
    derivatives.eth2_V = s(PrimitiveScratch::Eth2V);
    derivatives.ethprime2_Csharp =
        s(PrimitiveScratch::EthPrime2CSharp);
    derivatives.eth2_C = s(PrimitiveScratch::Eth2C);
    derivatives.capital_delta2_Csharp =
        s(PrimitiveScratch::Delta2CSharp);
    derivatives.capital_delta2_C = s(PrimitiveScratch::Delta2C);
    derivatives.capital_delta2_V = s(PrimitiveScratch::Delta2V);
    derivatives.capital_delta2_Vsharp =
        s(PrimitiveScratch::Delta2VSharp);
    derivatives.eth1_B = s(PrimitiveScratch::Eth1B);
    derivatives.ethprime1_Bsharp =
        s(PrimitiveScratch::EthPrime1BSharp);
    derivatives.thorn2_Sig = thorn_sig;
    derivatives.eth3_Kap = s(PrimitiveScratch::Eth3Kap);
    derivatives.thorn2_Be = thorn_be;
    derivatives.eth2_Ep = s(PrimitiveScratch::Eth2Ep);
    derivatives.capital_delta1_beta0 =
        background.capital_delta1_beta0;
    derivatives.capital_delta2_epsilon0 =
        background.capital_delta2_epsilon0;
    derivatives.bardelta2_epsilon0 = background.bardelta2_epsilon0;
    // The point evaluator returns the two regular remainders when these
    // explicit quotient slots are zero.  The positive-node extraction below
    // remains the sole authority for the quotients.
    derivatives.psi0_leading_combination_over_r2 = Complex{};
    derivatives.psi1_leading_combination_over_r = Complex{};
    const auto primitives =
        plus2_source_primitives(radius, background, metric, derivatives);
    const Complex values[4]{primitives.psi0_leading,
                            primitives.psi1_leading, primitives.Z0,
                            primitives.Z1};
    const std::size_t offset = 4 * time_level;
    bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < 4; ++field) {
      profiles[flat4(mode, offset + field, radial, theta, profile_count,
                     radial_count, theta_count)] = values[field];
      valid = valid && Kokkos::isfinite(values[field].real()) &&
              Kokkos::isfinite(values[field].imag());
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct ComputeCurvatureFunctor {
  UniformRadialGrid grid;
  const Complex* profiles;
  Complex* curvature;
  Complex* endpoint_audit;
  std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t profile_count =
        static_cast<std::size_t>(Profile::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t audit_count = 9;
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const double radius = grid.coordinate(radial);
    bool valid = ready[0] != 0;
    for (std::size_t level = 0; level < 3; ++level) {
      const std::size_t profile_offset = 4 * level;
      const std::size_t f0_index = flat4(
          mode, profile_offset, 0, theta, profile_count, radial_count,
          theta_count);
      const std::size_t f1_index = flat4(
          mode, profile_offset + 1, 0, theta, profile_count, radial_count,
          theta_count);
      const Complex* f0 = profiles + f0_index;
      const Complex* f1 = profiles + f1_index;
      const Complex q0 =
          radius == 0.0
              ? plus2_extract_q0_promoted_at_scri(
                    f0, radial_count, 1.0 / grid.spacing(), theta_count)
              : profiles[flat4(mode, profile_offset, radial, theta,
                               profile_count, radial_count, theta_count)] /
                    (radius * radius);
      const Complex q1 =
          radius == 0.0
              ? plus2_extract_q1_at_scri(f1, radial_count,
                                         1.0 / grid.spacing(), theta_count)
              : profiles[flat4(mode, profile_offset + 1, radial, theta,
                               profile_count, radial_count, theta_count)] /
                    radius;
      const Complex z0 =
          q0 + profiles[flat4(mode, profile_offset + 2, radial, theta,
                              profile_count, radial_count, theta_count)];
      const Complex z1 =
          q1 + profiles[flat4(mode, profile_offset + 3, radial, theta,
                              profile_count, radial_count, theta_count)];
      const std::size_t output_offset = 2 * level;
      curvature[flat4(mode, output_offset, radial, theta, curvature_count,
                      radial_count, theta_count)] = z0;
      curvature[flat4(mode, output_offset + 1, radial, theta,
                      curvature_count, radial_count, theta_count)] = z1;
      valid = valid && Kokkos::isfinite(z0.real()) &&
              Kokkos::isfinite(z0.imag()) && Kokkos::isfinite(z1.real()) &&
              Kokkos::isfinite(z1.imag());
      if (radial == 0) {
        const auto residuals = plus2_peeling_residuals_at_scri(
            f0, f1, radial_count, 1.0 / grid.spacing(), theta_count);
        const Complex audit[3]{residuals.f0_constant, residuals.f0_linear,
                               residuals.f1_constant};
        for (std::size_t component = 0; component < 3; ++component) {
          endpoint_audit[(mode * audit_count + 3 * level + component) *
                             theta_count +
                         theta] = audit[component];
          valid = valid && Kokkos::isfinite(audit[component].real()) &&
                  Kokkos::isfinite(audit[component].imag());
        }
      }
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct CurvatureRadialFunctor {
  const Complex* curvature;
  Complex* radial;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t radial_count_fields =
        static_cast<std::size_t>(CurvatureRadial::Count);
    constexpr std::size_t input_fields[4]{0, 1, 2, 3};
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % 4;
    const std::size_t mode = mode_field / 4;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta = within - radial_index * theta_count;
    const std::size_t input = flat4(mode, input_fields[field], 0, theta,
                                    curvature_count, radial_count,
                                    theta_count);
    radial[flat4(mode, field, radial_index, theta, radial_count_fields,
                 radial_count, theta_count)] =
        routeb_fornberg_direct_derivative_at(
            1, curvature + input, radial_count, radial_index, inverse_spacing,
            theta_count);
  }
};

struct ComputeDeltaDerivativesFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Complex* curvature;
  const Complex* radial;
  Complex* derivatives;
  std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t radial_fields =
        static_cast<std::size_t>(CurvatureRadial::Count);
    constexpr std::size_t derivative_count = static_cast<std::size_t>(
        Plus2BianchiDerivativeComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta = within - radial_index * theta_count;
    const auto c = [&](const Plus2TransportedCurvatureComponent field) {
      return curvature[flat4(mode, static_cast<std::size_t>(field),
                             radial_index, theta, curvature_count,
                             radial_count, theta_count)];
    };
    const auto dr = [&](const CurvatureRadial field) {
      return radial[flat4(mode, static_cast<std::size_t>(field), radial_index,
                          theta, radial_fields, radial_count, theta_count)];
    };
    const double radius = grid.coordinate(radial_index);
    const Complex delta4_z1 = delta_n_point(
        c(Plus2TransportedCurvatureComponent::Z1),
        c(Plus2TransportedCurvatureComponent::Z1T),
        dr(CurvatureRadial::DrZ1), 4, radius, parameters.mass,
        parameters.compactification_length);
    const Complex delta4_z1t = delta_n_point(
        c(Plus2TransportedCurvatureComponent::Z1T),
        c(Plus2TransportedCurvatureComponent::Z1TT),
        dr(CurvatureRadial::DrZ1T), 4, radius, parameters.mass,
        parameters.compactification_length);
    const Complex delta5_z0 = delta_n_point(
        c(Plus2TransportedCurvatureComponent::Z0),
        c(Plus2TransportedCurvatureComponent::Z0T),
        dr(CurvatureRadial::DrZ0), 5, radius, parameters.mass,
        parameters.compactification_length);
    const Complex delta5_z0t = delta_n_point(
        c(Plus2TransportedCurvatureComponent::Z0T),
        c(Plus2TransportedCurvatureComponent::Z0TT),
        dr(CurvatureRadial::DrZ0T), 5, radius, parameters.mass,
        parameters.compactification_length);
    const Complex values[4]{delta4_z1, delta4_z1t, delta5_z0, delta5_z0t};
    constexpr std::size_t slots[4]{0, 1, 4, 5};
    bool valid = ready[0] != 0;
    for (std::size_t i = 0; i < 4; ++i) {
      derivatives[flat4(mode, slots[i], radial_index, theta,
                        derivative_count, radial_count, theta_count)] =
          values[i];
      valid = valid && Kokkos::isfinite(values[i].real()) &&
              Kokkos::isfinite(values[i].imag());
    }
    constexpr std::size_t angular_slots[4]{2, 3, 6, 7};
    for (const std::size_t slot : angular_slots) {
      const Complex value = derivatives[flat4(
          mode, slot, radial_index, theta, derivative_count, radial_count,
          theta_count)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag());
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct FinalizeFunctor {
  Complex* curvature;
  Complex* derivatives;
  Complex* endpoint_audit;
  std::uint64_t* curvature_stamps;
  std::uint64_t* derivative_stamps;
  std::uint64_t* audit_stamps;
  const std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t derivative_count = static_cast<std::size_t>(
        Plus2BianchiDerivativeComponent::Count);
    constexpr std::size_t audit_count = 9;
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < curvature_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      curvature_count, radial_count,
                                      theta_count);
      if (!valid) curvature[index] = Complex{};
      curvature_stamps[index] = valid ? generation : 0;
    }
    for (std::size_t field = 0; field < derivative_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      derivative_count, radial_count,
                                      theta_count);
      if (!valid) derivatives[index] = Complex{};
      derivative_stamps[index] = valid ? generation : 0;
    }
    if (radial == 0) {
      for (std::size_t field = 0; field < audit_count; ++field) {
        const std::size_t index =
            (mode * audit_count + field) * theta_count + theta;
        if (!valid) endpoint_audit[index] = Complex{};
        audit_stamps[index] = valid ? generation : 0;
      }
    }
  }
};

static_assert(std::is_trivially_copyable_v<SetReadyFunctor>);
static_assert(std::is_trivially_copyable_v<ValidateTowerFunctor>);
static_assert(std::is_trivially_copyable_v<MetricRadialFunctor>);
static_assert(std::is_trivially_copyable_v<ConnectionRadialFunctor>);
static_assert(std::is_trivially_copyable_v<CurvatureProfileFunctor>);
static_assert(std::is_trivially_copyable_v<ComputeCurvatureFunctor>);
static_assert(std::is_trivially_copyable_v<CurvatureRadialFunctor>);
static_assert(std::is_trivially_copyable_v<ComputeDeltaDerivativesFunctor>);
static_assert(std::is_trivially_copyable_v<FinalizeFunctor>);

}  // namespace plus2_routeb_curvature_detail

// Standalone, allocation-free hot provider for all six local Route-B
// curvature fields and the eight derivative slots consumed by the live
// source.  It deliberately consumes an already-qualified h[0]..h[4] tower;
// it neither evolves Route A nor calls the nonlinear source graph.
template <class ExecSpace = ExecutionSpace>
class Plus2RouteBCurvatureSpatialProvider {
 public:
  using execution_space = ExecSpace;
  using scratch_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
  using stamp_view =
      Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
  using audit_view =
      Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
  using audit_stamp_view =
      Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, MemorySpace>;

  Plus2RouteBCurvatureSpatialProvider(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& grid, const KerrParameters& parameters,
      const int ell_max, const Plus2SpatialThetaView& cos_theta,
      const Plus2SpatialThetaView& sin_theta,
      const std::string& label = "plus2_routeb_curvature_spatial")
      : registry_(registry),
        grid_(grid),
        parameters_(parameters),
        ell_max_(ell_max),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        sharp_(label + "_sharp", registry.size()),
        modes_(label + "_modes", registry.size()),
        radius_(label + "_radius", grid.size()),
        primitive_scratch_(
            label + "_primitive_scratch", registry.size(),
            static_cast<std::size_t>(
                plus2_primitive_spatial_detail::Scratch::Count),
            grid.size(), sin_theta.extent(0)),
        primitive_value_(label + "_primitive_value", registry.size(),
                         static_cast<std::size_t>(
                             Plus2SpatialPrimitive::Count),
                         grid.size(), sin_theta.extent(0)),
        primitive_tangent_(label + "_primitive_tangent", registry.size(),
                           static_cast<std::size_t>(
                               Plus2SpatialPrimitive::Count),
                           grid.size(), sin_theta.extent(0)),
        jk_value_(label + "_jk_value", registry.size(),
                  static_cast<std::size_t>(
                      Plus2ProductionJkDerivative::Count),
                  grid.size(), sin_theta.extent(0)),
        jk_tangent_(label + "_jk_tangent", registry.size(),
                    static_cast<std::size_t>(
                        Plus2ProductionJkDerivative::Count),
                    grid.size(), sin_theta.extent(0)),
        profiles_(label + "_profiles", registry.size(),
                  static_cast<std::size_t>(
                      plus2_routeb_curvature_detail::Profile::Count),
                  grid.size(), sin_theta.extent(0)),
        curvature_(label + "_curvature", registry.size(),
                   static_cast<std::size_t>(
                       Plus2TransportedCurvatureComponent::Count),
                   grid.size(), sin_theta.extent(0)),
        curvature_stamps_(label + "_curvature_stamps", registry.size(),
                          static_cast<std::size_t>(
                              Plus2TransportedCurvatureComponent::Count),
                          grid.size(), sin_theta.extent(0)),
        derivatives_(label + "_derivatives", registry.size(),
                     static_cast<std::size_t>(
                         Plus2BianchiDerivativeComponent::Count),
                     grid.size(), sin_theta.extent(0)),
        derivative_stamps_(label + "_derivative_stamps", registry.size(),
                           static_cast<std::size_t>(
                               Plus2BianchiDerivativeComponent::Count),
                           grid.size(), sin_theta.extent(0)),
        curvature_radial_(
            label + "_curvature_radial", registry.size(),
            static_cast<std::size_t>(
                plus2_routeb_curvature_detail::CurvatureRadial::Count),
            grid.size(), sin_theta.extent(0)),
        endpoint_audit_(label + "_endpoint_audit", registry.size(), 9,
                        sin_theta.extent(0)),
        endpoint_audit_stamps_(label + "_endpoint_audit_stamps",
                               registry.size(), 9, sin_theta.extent(0)),
        ready_(label + "_ready", 1),
        bsharp_angular_(execution, registry, 2, 0, ell_max,
                        static_cast<int>(sin_theta.extent(0)), grid.size(),
                        parameters),
        csharp_angular_(execution, registry, 1, 1, ell_max,
                        static_cast<int>(sin_theta.extent(0)), grid.size(),
                        parameters),
        v_angular_(execution, registry, 0, 2, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters),
        c_angular_(execution, registry, -1, 1, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters),
        b_angular_(execution, registry, -2, 0, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters),
        kap_angular_(execution, registry, 1, 2, ell_max,
                     static_cast<int>(sin_theta.extent(0)), grid.size(),
                     parameters),
        ep_angular_(execution, registry, 0, 1, ell_max,
                    static_cast<int>(sin_theta.extent(0)), grid.size(),
                    parameters),
        z1_angular_(execution, registry, 1, 1, ell_max,
                    static_cast<int>(sin_theta.extent(0)), grid.size(),
                    parameters),
        z0_angular_(execution, registry, 2, 2, ell_max,
                    static_cast<int>(sin_theta.extent(0)), grid.size(),
                    parameters) {
    if (!registry.is_closed_under_sharp() || registry.size() == 0 ||
        grid.lower_radius() != 0.0 ||
        grid.size() < routeb_fornberg_window ||
        ell_max < 2 || sin_theta.extent(0) == 0 ||
        sin_theta.extent(0) != cos_theta.extent(0) ||
        !std::isfinite(parameters.mass) ||
        !std::isfinite(parameters.spin) ||
        !std::isfinite(parameters.compactification_length) ||
        parameters.mass <= 0.0 ||
        std::abs(parameters.spin) > parameters.mass ||
        parameters.compactification_length <= 0.0) {
      throw std::invalid_argument(
          "Route-B curvature provider requires closed D10-5 Kerr geometry");
    }
    auto host_sharp = Kokkos::create_mirror_view(sharp_);
    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_radius = Kokkos::create_mirror_view(radius_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_sharp(mode) = registry.sharp_index(registry.modes()[mode]);
      host_modes(mode) = registry.modes()[mode];
    }
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      host_radius(radial) = grid.coordinate(radial);
    }
    Kokkos::deep_copy(execution, sharp_, host_sharp);
    Kokkos::deep_copy(execution, modes_, host_modes);
    Kokkos::deep_copy(execution, radius_, host_radius);
    Kokkos::deep_copy(execution, curvature_, Complex{});
    Kokkos::deep_copy(execution, derivatives_, Complex{});
    Kokkos::deep_copy(execution, curvature_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, derivative_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, endpoint_audit_, Complex{});
    Kokkos::deep_copy(execution, endpoint_audit_stamps_, std::uint64_t{0});
  }

  [[nodiscard]] Plus2TransportedCurvatureStage curvature_stage() const {
    return {curvature_, curvature_stamps_};
  }
  [[nodiscard]] Plus2BianchiDerivativeStage derivative_stage() const {
    return {derivatives_, derivative_stamps_};
  }
  [[nodiscard]] audit_view endpoint_audit() const { return endpoint_audit_; }
  [[nodiscard]] audit_stamp_view endpoint_audit_stamps() const {
    return endpoint_audit_stamps_;
  }
  [[nodiscard]] Kokkos::View<const std::uint8_t*, MemorySpace> readiness()
      const {
    return ready_;
  }
  [[nodiscard]] std::uint64_t last_generation() const noexcept {
    return last_generation_;
  }

  void evaluate(
      const execution_space& execution,
      const Plus2RouteBCurvatureTowerStage& tower,
      const Plus2RouteBCurvatureOffsets offsets = {}) {
    validate(tower, offsets);
    using namespace plus2_routeb_curvature_detail;
    using namespace plus2_primitive_spatial_detail;
    const std::size_t mode_count = registry_.size();
    const std::size_t radial_count = grid_.size();
    const std::size_t theta_count = sin_theta_.extent(0);
    const std::size_t points = mode_count * radial_count * theta_count;
    Kokkos::parallel_for(
        "plus2_routeb_curvature_set_ready",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        plus2_routeb_curvature_detail::SetReadyFunctor{ready_.data()});
    Kokkos::parallel_for(
        "plus2_routeb_curvature_validate_tower",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        ValidateTowerFunctor{
            tower.fields.data(), tower.stamps.data(), ready_.data(),
            tower.generation, mode_count, tower.fields.extent(2), radial_count,
            theta_count, offsets.H, offsets.B, offsets.Pi, offsets.C,
            offsets.U});
    const std::size_t level_stride = tower.fields.stride(0);
    for (std::size_t level = 0; level < 3; ++level) {
      const Complex* value = tower.fields.data() + level * level_stride;
      const Complex* tangent =
          tower.fields.data() + (level + 1) * level_stride;
      const Complex* second =
          tower.fields.data() + (level + 2) * level_stride;
      Kokkos::parallel_for(
          "plus2_routeb_curvature_pack",
          Kokkos::RangePolicy<execution_space>(execution, 0, points),
          PackFunctor{grid_, parameters_, cos_theta_.data(),
                      sin_theta_.data(), value, tangent, second, sharp_.data(),
                      ready_.data(), primitive_scratch_.data(),
                      tower.fields.extent(2), radial_count, theta_count,
                      offsets.H, offsets.Pi, offsets.B, offsets.C, offsets.U});
      Kokkos::parallel_for(
          "plus2_routeb_curvature_metric_radial",
          Kokkos::RangePolicy<execution_space>(
              execution, 0, mode_count * 10 * radial_count * theta_count),
          plus2_routeb_curvature_detail::MetricRadialFunctor{
              primitive_scratch_.data(), primitive_scratch_.data(),
              radial_count, theta_count, 1.0 / grid_.spacing()});
      apply_metric_angular(execution);
      Kokkos::parallel_for(
          "plus2_routeb_curvature_metric_point",
          Kokkos::RangePolicy<execution_space>(execution, 0, points),
          MetricPointFunctor{
              grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
              modes_.data(), primitive_scratch_.data(),
              primitive_scratch_.data(), primitive_value_.data(),
              primitive_tangent_.data(), jk_value_.data(),
              jk_tangent_.data(), radial_count, theta_count});
      Kokkos::parallel_for(
          "plus2_routeb_curvature_connection_radial",
          Kokkos::RangePolicy<execution_space>(
              execution, 0, mode_count * 3 * radial_count * theta_count),
          plus2_routeb_curvature_detail::ConnectionRadialFunctor{
              primitive_scratch_.data(), primitive_scratch_.data(),
              radial_count, theta_count, 1.0 / grid_.spacing()});
      kap_angular_.eth(
          execution, primitive_scratch_,
          static_cast<std::size_t>(PrimitiveScratch::Kap),
          primitive_scratch_,
          static_cast<std::size_t>(PrimitiveScratch::KapT), radius_,
          sin_theta_, cos_theta_, primitive_scratch_,
          static_cast<std::size_t>(PrimitiveScratch::Eth3Kap));
      ep_angular_.eth(execution, primitive_scratch_,
                      static_cast<std::size_t>(PrimitiveScratch::Ep),
                      primitive_scratch_,
                      static_cast<std::size_t>(PrimitiveScratch::EpT),
                      radius_, sin_theta_, cos_theta_, primitive_scratch_,
                      static_cast<std::size_t>(PrimitiveScratch::Eth2Ep));
      Kokkos::parallel_for(
          "plus2_routeb_curvature_profiles",
          Kokkos::RangePolicy<execution_space>(execution, 0, points),
          CurvatureProfileFunctor{
              grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
              modes_.data(), primitive_scratch_.data(), profiles_.data(),
              ready_.data(), radial_count, theta_count, level});
    }
    Kokkos::parallel_for(
        "plus2_routeb_curvature_extract",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        ComputeCurvatureFunctor{
            grid_, profiles_.data(), curvature_.data(),
            endpoint_audit_.data(), ready_.data(), radial_count,
            theta_count});
    Kokkos::parallel_for(
        "plus2_routeb_curvature_radial",
        Kokkos::RangePolicy<execution_space>(
            execution, 0, mode_count * 4 * radial_count * theta_count),
        CurvatureRadialFunctor{curvature_.data(), curvature_radial_.data(),
                               radial_count, theta_count,
                               1.0 / grid_.spacing()});
    z1_angular_.ethprime(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        radius_, sin_theta_, cos_theta_, derivatives_,
        static_cast<std::size_t>(
            Plus2BianchiDerivativeComponent::EthPrime4Z1));
    z1_angular_.ethprime(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1T),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z1TT),
        radius_, sin_theta_, cos_theta_, derivatives_,
        static_cast<std::size_t>(
            Plus2BianchiDerivativeComponent::EthPrime4Z1T));
    z0_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0T),
        radius_, sin_theta_, cos_theta_, derivatives_,
        static_cast<std::size_t>(Plus2BianchiDerivativeComponent::Eth5Z0));
    z0_angular_.eth(
        execution, curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0T),
        curvature_,
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Z0TT),
        radius_, sin_theta_, cos_theta_, derivatives_,
        static_cast<std::size_t>(Plus2BianchiDerivativeComponent::Eth5Z0T));
    Kokkos::parallel_for(
        "plus2_routeb_curvature_delta_derivatives",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        ComputeDeltaDerivativesFunctor{
            grid_, parameters_, curvature_.data(), curvature_radial_.data(),
            derivatives_.data(), ready_.data(), radial_count, theta_count});
    Kokkos::parallel_for(
        "plus2_routeb_curvature_finalize",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        plus2_routeb_curvature_detail::FinalizeFunctor{
            curvature_.data(), derivatives_.data(), endpoint_audit_.data(),
            curvature_stamps_.data(), derivative_stamps_.data(),
            endpoint_audit_stamps_.data(), ready_.data(), tower.generation,
            radial_count, theta_count});
    last_generation_ = tower.generation;
  }

 private:
  template <class ViewA, class ViewB>
  static bool overlaps(const ViewA& left, const ViewB& right) {
    if (left.data() == nullptr || right.data() == nullptr) return false;
    const auto left_begin =
        reinterpret_cast<std::uintptr_t>(left.data());
    const auto right_begin =
        reinterpret_cast<std::uintptr_t>(right.data());
    const auto left_end =
        left_begin + left.span() * sizeof(typename ViewA::non_const_value_type);
    const auto right_end = right_begin +
                           right.span() *
                               sizeof(typename ViewB::non_const_value_type);
    return left_begin < right_end && right_begin < left_end;
  }

  void validate(const Plus2RouteBCurvatureTowerStage& tower,
                const Plus2RouteBCurvatureOffsets& offsets) const {
    const std::size_t largest =
        std::max({offsets.H, offsets.B, offsets.Pi, offsets.C, offsets.U});
    const std::size_t selected[5]{offsets.H, offsets.B, offsets.Pi,
                                  offsets.C, offsets.U};
    bool unique_offsets = true;
    for (std::size_t left = 0; left < 5; ++left) {
      for (std::size_t right = left + 1; right < 5; ++right) {
        unique_offsets = unique_offsets && selected[left] != selected[right];
      }
    }
    const bool shapes =
        tower.fields.extent(0) == 5 &&
        tower.fields.extent(1) == registry_.size() &&
        tower.fields.extent(2) > largest &&
        tower.fields.extent(3) == grid_.size() &&
        tower.fields.extent(4) == sin_theta_.extent(0) &&
        tower.stamps.extent(0) == 5 &&
        tower.stamps.extent(1) == registry_.size() &&
        tower.stamps.extent(2) == grid_.size() &&
        tower.stamps.extent(3) == sin_theta_.extent(0);
    const bool aliases =
        overlaps(tower.fields, primitive_scratch_) ||
        overlaps(tower.fields, primitive_value_) ||
        overlaps(tower.fields, primitive_tangent_) ||
        overlaps(tower.fields, jk_value_) ||
        overlaps(tower.fields, jk_tangent_) ||
        overlaps(tower.fields, profiles_) ||
        overlaps(tower.fields, curvature_) ||
        overlaps(tower.fields, derivatives_) ||
        overlaps(tower.fields, curvature_radial_) ||
        overlaps(tower.fields, endpoint_audit_) ||
        overlaps(tower.fields, endpoint_audit_stamps_) ||
        overlaps(tower.fields, ready_) || overlaps(tower.fields, sharp_) ||
        overlaps(tower.fields, modes_) || overlaps(tower.fields, radius_) ||
        overlaps(tower.fields, cos_theta_) ||
        overlaps(tower.fields, sin_theta_) ||
        overlaps(tower.stamps, tower.fields) ||
        overlaps(tower.stamps, curvature_stamps_) ||
        overlaps(tower.stamps, derivative_stamps_) ||
        overlaps(tower.stamps, endpoint_audit_stamps_) ||
        overlaps(tower.stamps, endpoint_audit_) ||
        overlaps(tower.stamps, ready_) || overlaps(tower.stamps, sharp_) ||
        overlaps(tower.stamps, modes_) || overlaps(tower.stamps, radius_) ||
        overlaps(tower.stamps, cos_theta_) ||
        overlaps(tower.stamps, sin_theta_);
    if (tower.generation == 0 || tower.generation <= last_generation_ ||
        !shapes || !unique_offsets || aliases) {
      throw std::invalid_argument(
          "Route-B curvature tower contract is incomplete");
    }
  }

  void apply_metric_angular(const execution_space& execution) {
    using S = plus2_primitive_spatial_detail::Scratch;
    const auto eth = [&](auto& coordinator, const S value, const S tangent,
                         const S output) {
      coordinator.eth(execution, primitive_scratch_,
                      static_cast<std::size_t>(value), primitive_scratch_,
                      static_cast<std::size_t>(tangent), radius_, sin_theta_,
                      cos_theta_, primitive_scratch_,
                      static_cast<std::size_t>(output));
    };
    const auto ethprime = [&](auto& coordinator, const S value,
                              const S tangent, const S output) {
      coordinator.ethprime(
          execution, primitive_scratch_, static_cast<std::size_t>(value),
          primitive_scratch_, static_cast<std::size_t>(tangent), radius_,
          sin_theta_, cos_theta_, primitive_scratch_,
          static_cast<std::size_t>(output));
    };
    eth(v_angular_, S::V, S::VT, S::Eth2V);
    eth(v_angular_, S::VT, S::VTT, S::Eth2VT);
    ethprime(csharp_angular_, S::CSharp, S::CSharpT,
             S::EthPrime2CSharp);
    ethprime(csharp_angular_, S::CSharpT, S::CSharpTT,
             S::EthPrime2CSharpT);
    eth(c_angular_, S::C, S::CT, S::Eth2C);
    eth(c_angular_, S::CT, S::CTT, S::Eth2CT);
    eth(b_angular_, S::B, S::BT, S::Eth1B);
    eth(b_angular_, S::BT, S::BTT, S::Eth1BT);
    ethprime(bsharp_angular_, S::BSharp, S::BSharpT,
             S::EthPrime1BSharp);
    ethprime(bsharp_angular_, S::BSharpT, S::BSharpTT,
             S::EthPrime1BSharpT);
  }

  ModeRegistry registry_;
  UniformRadialGrid grid_;
  KerrParameters parameters_;
  int ell_max_;
  Plus2SpatialThetaView cos_theta_;
  Plus2SpatialThetaView sin_theta_;
  Kokkos::View<std::size_t*, MemorySpace> sharp_;
  Kokkos::View<int*, MemorySpace> modes_;
  Kokkos::View<Real*, MemorySpace> radius_;
  scratch_view primitive_scratch_;
  scratch_view primitive_value_;
  scratch_view primitive_tangent_;
  scratch_view jk_value_;
  scratch_view jk_tangent_;
  scratch_view profiles_;
  Plus2TransportedCurvatureStorageView curvature_;
  Plus2LiveStampView curvature_stamps_;
  Plus2BianchiDerivativeView derivatives_;
  Plus2LiveStampView derivative_stamps_;
  scratch_view curvature_radial_;
  audit_view endpoint_audit_;
  audit_stamp_view endpoint_audit_stamps_;
  Kokkos::View<std::uint8_t*, MemorySpace> ready_;
  SignedModeAngularCoordinator<execution_space> bsharp_angular_;
  SignedModeAngularCoordinator<execution_space> csharp_angular_;
  SignedModeAngularCoordinator<execution_space> v_angular_;
  SignedModeAngularCoordinator<execution_space> c_angular_;
  SignedModeAngularCoordinator<execution_space> b_angular_;
  SignedModeAngularCoordinator<execution_space> kap_angular_;
  SignedModeAngularCoordinator<execution_space> ep_angular_;
  SignedModeAngularCoordinator<execution_space> z1_angular_;
  SignedModeAngularCoordinator<execution_space> z0_angular_;
  std::uint64_t last_generation_ = 0;
};

}  // namespace teuk
