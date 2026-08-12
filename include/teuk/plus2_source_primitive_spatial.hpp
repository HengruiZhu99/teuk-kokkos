#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/angular_coordinator.hpp"
#include "teuk/ghp.hpp"
#include "teuk/grid.hpp"
#include "teuk/jet.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_source_primitives.hpp"
#include "teuk/plus2_source_value_spatial.hpp"
#include "teuk/plus2_transported_curvature.hpp"
#include "teuk/radial_discretization.hpp"

namespace teuk {

// Read-only common-stage reconstruction input.  Every h[0], h[1], and h[2]
// field has a pointwise generation stamp.  This is intentionally stricter
// than accepting three bare views: radial and angular operators couple points,
// so one stale input invalidates the complete producer invocation.
using Plus2PrimitiveConstStageView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2PrimitiveConstStampView =
    Kokkos::View<const std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;

struct Plus2PrimitiveReconstructionStage {
  std::uint64_t generation = 0;
  Plus2PrimitiveConstStageView value;
  Plus2PrimitiveConstStageView tangent;
  Plus2PrimitiveConstStageView second_tangent;
  Plus2PrimitiveConstStampView value_stamps;
  Plus2PrimitiveConstStampView tangent_stamps;
  Plus2PrimitiveConstStampView second_tangent_stamps;
};

struct Plus2PrimitiveReconstructionOffsets {
  std::size_t H = 0;
  std::size_t Pi = 1;
  std::size_t B = 2;
  std::size_t C = 3;
  std::size_t U = 4;
};

namespace plus2_primitive_spatial_detail {

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

enum class Scratch : std::size_t {
  H,
  HT,
  Pi,
  PiT,
  U,
  UT,
  UTT,
  USharp,
  USharpT,
  USharpTT,
  C,
  CT,
  CTT,
  CSharp,
  CSharpT,
  CSharpTT,
  B,
  BT,
  BTT,
  BSharp,
  BSharpT,
  BSharpTT,
  V,
  VT,
  VTT,
  VSharp,
  VSharpT,
  VSharpTT,
  DrBSharp,
  DrBSharpT,
  DrCSharp,
  DrCSharpT,
  DrC,
  DrCT,
  DrV,
  DrVT,
  DrVSharp,
  DrVSharpT,
  Thorn1BSharp,
  Thorn1BSharpT,
  Thorn2CSharp,
  Thorn2CSharpT,
  Eth2V,
  Eth2VT,
  EthPrime2CSharp,
  EthPrime2CSharpT,
  Eth2C,
  Eth2CT,
  Eth1B,
  Eth1BT,
  EthPrime1BSharp,
  EthPrime1BSharpT,
  Delta2CSharp,
  Delta2CSharpT,
  Delta2C,
  Delta2CT,
  Delta2V,
  Delta2VT,
  Delta2VSharp,
  Delta2VSharpT,
  Sig,
  SigT,
  Kap,
  KapT,
  DrSig,
  DrKap,
  EthPrime3Kap,
  Delta2Sig,
  Delta3Kap,
  Count,
};

template <class Scalar>
struct MetricFieldsT {
  Scalar U;
  Scalar Usharp;
  Scalar C;
  Scalar Csharp;
  Scalar B;
  Scalar Bsharp;
  Scalar H;
  Scalar Pi;
};

template <class Scalar>
struct MetricDerivativesT {
  Scalar thorn1_Bsharp;
  Scalar thorn2_Csharp;
  Scalar eth2_V;
  Scalar ethprime2_Csharp;
  Scalar eth2_C;
  Scalar capital_delta2_Csharp;
  Scalar capital_delta2_C;
  Scalar capital_delta2_V;
  Scalar capital_delta2_Vsharp;
  Scalar eth1_B;
  Scalar ethprime1_Bsharp;
};

// The twelve non-curvature manifest rows. Z0 and Z1 are deliberately absent:
// this layer consumes a typed curvature adapter and never reconstructs a
// quotient or invents a scri coefficient. Route A may supply that adapter only
// in validation; rotating production requires the local Route-B graph.
template <class Scalar>
struct ConnectionPrimitivesT {
  Scalar H;
  Scalar Sig;
  Scalar Kap;
  Scalar Rh;
  Scalar Ta;
  Scalar Al;
  Scalar Be;
  Scalar Ep;
  Scalar Pi;
  Scalar V;
  Scalar C;
  Scalar B;
};

template <class Scalar>
KOKKOS_INLINE_FUNCTION ConnectionPrimitivesT<Scalar>
connection_primitives(const double radius,
                      const Plus2PrimitiveBackground& background,
                      const MetricFieldsT<Scalar>& metric,
                      const MetricDerivativesT<Scalar>& d) {
  const Plus2ReconstructionPrimitiveInputsT<Scalar> fields{
      metric.U, metric.Usharp, metric.C, metric.Csharp,
      metric.B, metric.Bsharp, metric.H, metric.Pi};
  Plus2PrimitiveDerivativesT<Scalar> derivatives{};
  derivatives.thorn1_Bsharp = d.thorn1_Bsharp;
  derivatives.thorn2_Csharp = d.thorn2_Csharp;
  derivatives.eth2_V = d.eth2_V;
  derivatives.ethprime2_Csharp = d.ethprime2_Csharp;
  derivatives.eth2_C = d.eth2_C;
  derivatives.capital_delta2_Csharp = d.capital_delta2_Csharp;
  derivatives.capital_delta2_C = d.capital_delta2_C;
  derivatives.capital_delta2_V = d.capital_delta2_V;
  derivatives.capital_delta2_Vsharp = d.capital_delta2_Vsharp;
  derivatives.eth1_B = d.eth1_B;
  derivatives.ethprime1_Bsharp = d.ethprime1_Bsharp;
  // Curvature-quotient derivative slots enter only Z0/Z1. Those returned
  // values are ignored in favor of the typed curvature adapter.
  const auto p = plus2_source_primitives(radius, background, fields,
                                         derivatives);
  return {p.H, p.Sig, p.Kap, p.Rh, p.Ta, p.Al,
          p.Be, p.Ep,  p.Pi,  p.V,  p.C,  p.B};
}

struct SetReadyFunctor {
  std::uint8_t* ready;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t) const {
    ready[0] = static_cast<std::uint8_t>(1);
  }
};

struct ValidateFunctor {
  const Complex* value;
  const Complex* tangent;
  const Complex* second;
  const std::uint64_t* value_stamps;
  const std::uint64_t* tangent_stamps;
  const std::uint64_t* second_stamps;
  const Complex* curvature;
  const std::uint64_t* curvature_stamps;
  const Complex* bianchi;
  const std::uint64_t* bianchi_stamps;
  std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t input_fields;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t H;
  std::size_t Pi;
  std::size_t B;
  std::size_t C;
  std::size_t U;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t bianchi_count = static_cast<std::size_t>(
        Plus2BianchiDerivativeComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t fields[5]{H, Pi, B, C, U};
    bool valid = generation != 0;
    for (const std::size_t field : fields) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      input_fields, radial_count, theta_count);
      const Complex values[3]{value[index], tangent[index], second[index]};
      valid = valid && value_stamps[index] == generation &&
              tangent_stamps[index] == generation &&
              second_stamps[index] == generation;
      for (const Complex x : values) {
        valid = valid && Kokkos::isfinite(x.real()) &&
                Kokkos::isfinite(x.imag());
      }
    }
    for (std::size_t field = 0; field < curvature_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      curvature_count, radial_count,
                                      theta_count);
      const Complex x = curvature[index];
      valid = valid && curvature_stamps[index] == generation &&
              Kokkos::isfinite(x.real()) && Kokkos::isfinite(x.imag());
    }
    for (std::size_t field = 0; field < bianchi_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      bianchi_count, radial_count,
                                      theta_count);
      const Complex x = bianchi[index];
      valid = valid && bianchi_stamps[index] == generation &&
              Kokkos::isfinite(x.real()) && Kokkos::isfinite(x.imag());
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct PackFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* value;
  const Complex* tangent;
  const Complex* second;
  const std::size_t* sharp;
  const std::uint8_t* ready;
  Complex* scratch;
  std::size_t input_fields;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t H;
  std::size_t Pi;
  std::size_t B;
  std::size_t C;
  std::size_t U;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t count = static_cast<std::size_t>(Scratch::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t sharp_mode = sharp[mode];
    const auto input = [&](const Complex* data, const std::size_t input_mode,
                           const std::size_t field) {
      return ready[0]
                 ? data[flat4(input_mode, field, radial, theta, input_fields,
                              radial_count, theta_count)]
                 : Complex{};
    };
    const auto put = [&](const Scratch field, const Complex x) {
      scratch[flat4(mode, static_cast<std::size_t>(field), radial, theta,
                    count, radial_count, theta_count)] = x;
    };
    const double radius = grid.coordinate(radial);
    const auto background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    const Complex u = input(value, mode, U);
    const Complex ut = input(tangent, mode, U);
    const Complex utt = input(second, mode, U);
    const Complex us = Kokkos::conj(input(value, sharp_mode, U));
    const Complex ust = Kokkos::conj(input(tangent, sharp_mode, U));
    const Complex ustt = Kokkos::conj(input(second, sharp_mode, U));
    put(Scratch::H, input(value, mode, H));
    put(Scratch::HT, input(tangent, mode, H));
    put(Scratch::Pi, input(value, mode, Pi));
    put(Scratch::PiT, input(tangent, mode, Pi));
    put(Scratch::U, u);
    put(Scratch::UT, ut);
    put(Scratch::UTT, utt);
    put(Scratch::USharp, us);
    put(Scratch::USharpT, ust);
    put(Scratch::USharpTT, ustt);
    put(Scratch::C, input(value, mode, C));
    put(Scratch::CT, input(tangent, mode, C));
    put(Scratch::CTT, input(second, mode, C));
    put(Scratch::CSharp, Kokkos::conj(input(value, sharp_mode, C)));
    put(Scratch::CSharpT, Kokkos::conj(input(tangent, sharp_mode, C)));
    put(Scratch::CSharpTT, Kokkos::conj(input(second, sharp_mode, C)));
    put(Scratch::B, input(value, mode, B));
    put(Scratch::BT, input(tangent, mode, B));
    put(Scratch::BTT, input(second, mode, B));
    put(Scratch::BSharp, Kokkos::conj(input(value, sharp_mode, B)));
    put(Scratch::BSharpT, Kokkos::conj(input(tangent, sharp_mode, B)));
    put(Scratch::BSharpTT, Kokkos::conj(input(second, sharp_mode, B)));
    put(Scratch::V, u / background.mu0);
    put(Scratch::VT, ut / background.mu0);
    put(Scratch::VTT, utt / background.mu0);
    put(Scratch::VSharp, us / Kokkos::conj(background.mu0));
    put(Scratch::VSharpT, ust / Kokkos::conj(background.mu0));
    put(Scratch::VSharpTT, ustt / Kokkos::conj(background.mu0));
  }
};

struct RadialMetricFunctor {
  const Complex* scratch_input;
  Complex* scratch_output;
  RadialDiscretization discretization;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t count = static_cast<std::size_t>(Scratch::Count);
    constexpr Scratch inputs[10]{
        Scratch::BSharp, Scratch::BSharpT, Scratch::CSharp,
        Scratch::CSharpT, Scratch::C, Scratch::CT, Scratch::V,
        Scratch::VT, Scratch::VSharp, Scratch::VSharpT};
    constexpr Scratch outputs[10]{
        Scratch::DrBSharp, Scratch::DrBSharpT, Scratch::DrCSharp,
        Scratch::DrCSharpT, Scratch::DrC, Scratch::DrCT, Scratch::DrV,
        Scratch::DrVT, Scratch::DrVSharp, Scratch::DrVSharpT};
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % 10;
    const std::size_t mode = mode_field / 10;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t base = flat4(
        mode, static_cast<std::size_t>(inputs[field]), 0, theta, count,
        radial_count, theta_count);
    scratch_output[flat4(mode, static_cast<std::size_t>(outputs[field]),
                         radial, theta, count, radial_count, theta_count)] =
        radial_first_derivative_strided_at(
            discretization, scratch_input + base, radial_count, radial,
            inverse_spacing, theta_count);
  }
};

struct MetricPointFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const int* modes;
  const Complex* scratch_input;
  Complex* scratch_output;
  Complex* primitive_value;
  Complex* primitive_tangent;
  Complex* jk_value;
  Complex* jk_tangent;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t sc = static_cast<std::size_t>(Scratch::Count);
    constexpr std::size_t pc =
        static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
    constexpr std::size_t dc =
        static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto s = [&](const Scratch field) {
      return scratch_input[flat4(mode, static_cast<std::size_t>(field),
                                 radial, theta, sc, radial_count,
                                 theta_count)];
    };
    const auto put_s = [&](const Scratch field, const Complex x) {
      scratch_output[flat4(mode, static_cast<std::size_t>(field), radial,
                           theta, sc, radial_count, theta_count)] = x;
    };
    const auto jet = [&](const Scratch value, const Scratch tangent) {
      return Jet1<Complex>{s(value), s(tangent)};
    };
    const double radius = grid.coordinate(radial);
    const auto background = plus2_primitive_background(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    const auto thorn = [&](const Scratch value, const Scratch tangent,
                           const Scratch radial_value,
                           const Scratch radial_tangent,
                           const Scratch second_tangent,
                           const int falloff, const int spin,
                           const int boost) {
      return Jet1<Complex>{
          thorn_n_point(s(value), s(tangent), s(radial_value), falloff, spin,
                        boost, modes[mode], radius, cos_theta[theta],
                        parameters.mass, parameters.spin,
                        parameters.compactification_length,
                        background.kerr.epsilon0),
          thorn_n_point(s(tangent), s(second_tangent), s(radial_tangent),
                        falloff, spin, boost, modes[mode], radius,
                        cos_theta[theta], parameters.mass, parameters.spin,
                        parameters.compactification_length,
                        background.kerr.epsilon0)};
    };
    const auto delta = [&](const Scratch value, const Scratch tangent,
                           const Scratch radial_value,
                           const Scratch radial_tangent,
                           const Scratch second_tangent,
                           const int falloff) {
      return Jet1<Complex>{
          delta_n_point(s(value), s(tangent), s(radial_value), falloff,
                        radius, parameters.mass,
                        parameters.compactification_length),
          delta_n_point(s(tangent), s(second_tangent), s(radial_tangent),
                        falloff, radius, parameters.mass,
                        parameters.compactification_length)};
    };
    const Jet1<Complex> thorn_bsharp =
        thorn(Scratch::BSharp, Scratch::BSharpT, Scratch::DrBSharp,
              Scratch::DrBSharpT, Scratch::BSharpTT, 1, 2, 0);
    const Jet1<Complex> thorn_csharp =
        thorn(Scratch::CSharp, Scratch::CSharpT, Scratch::DrCSharp,
              Scratch::DrCSharpT, Scratch::CSharpTT, 2, 1, 1);
    const Jet1<Complex> delta_csharp =
        delta(Scratch::CSharp, Scratch::CSharpT, Scratch::DrCSharp,
              Scratch::DrCSharpT, Scratch::CSharpTT, 2);
    const Jet1<Complex> delta_c =
        delta(Scratch::C, Scratch::CT, Scratch::DrC, Scratch::DrCT,
              Scratch::CTT, 2);
    const Jet1<Complex> delta_v =
        delta(Scratch::V, Scratch::VT, Scratch::DrV, Scratch::DrVT,
              Scratch::VTT, 2);
    const Jet1<Complex> delta_vsharp =
        delta(Scratch::VSharp, Scratch::VSharpT, Scratch::DrVSharp,
              Scratch::DrVSharpT, Scratch::VSharpTT, 2);
    put_s(Scratch::Thorn1BSharp, thorn_bsharp.value);
    put_s(Scratch::Thorn1BSharpT, thorn_bsharp.dt);
    put_s(Scratch::Thorn2CSharp, thorn_csharp.value);
    put_s(Scratch::Thorn2CSharpT, thorn_csharp.dt);
    const MetricFieldsT<Jet1<Complex>> fields{
        jet(Scratch::U, Scratch::UT),
        jet(Scratch::USharp, Scratch::USharpT),
        jet(Scratch::C, Scratch::CT),
        jet(Scratch::CSharp, Scratch::CSharpT),
        jet(Scratch::B, Scratch::BT),
        jet(Scratch::BSharp, Scratch::BSharpT),
        jet(Scratch::H, Scratch::HT),
        jet(Scratch::Pi, Scratch::PiT)};
    const MetricDerivativesT<Jet1<Complex>> derivatives{
        thorn_bsharp,
        thorn_csharp,
        jet(Scratch::Eth2V, Scratch::Eth2VT),
        jet(Scratch::EthPrime2CSharp, Scratch::EthPrime2CSharpT),
        jet(Scratch::Eth2C, Scratch::Eth2CT),
        delta_csharp,
        delta_c,
        delta_v,
        delta_vsharp,
        jet(Scratch::Eth1B, Scratch::Eth1BT),
        jet(Scratch::EthPrime1BSharp, Scratch::EthPrime1BSharpT)};
    const auto p = connection_primitives(radius, background, fields,
                                         derivatives);
    const Jet1<Complex> primitives[12]{
        p.H, p.Sig, p.Kap, p.Rh, p.Ta, p.Al,
        p.Be, p.Ep, p.Pi, p.V, p.C, p.B};
    constexpr Plus2SpatialPrimitive primitive_slots[12]{
        Plus2SpatialPrimitive::H, Plus2SpatialPrimitive::Sig,
        Plus2SpatialPrimitive::Kap, Plus2SpatialPrimitive::Rh,
        Plus2SpatialPrimitive::Ta, Plus2SpatialPrimitive::Al,
        Plus2SpatialPrimitive::Be, Plus2SpatialPrimitive::Ep,
        Plus2SpatialPrimitive::Pi, Plus2SpatialPrimitive::V,
        Plus2SpatialPrimitive::C, Plus2SpatialPrimitive::B};
    for (std::size_t i = 0; i < 12; ++i) {
      const std::size_t index = flat4(
          mode, static_cast<std::size_t>(primitive_slots[i]), radial, theta,
          pc, radial_count, theta_count);
      primitive_value[index] = primitives[i].value;
      primitive_tangent[index] = primitives[i].dt;
    }
    put_s(Scratch::Sig, p.Sig.value);
    put_s(Scratch::SigT, p.Sig.dt);
    put_s(Scratch::Kap, p.Kap.value);
    put_s(Scratch::KapT, p.Kap.dt);
    const Jet1<Complex> metric_derivatives[7]{
        delta_csharp,
        jet(Scratch::EthPrime1BSharp, Scratch::EthPrime1BSharpT),
        delta_c,
        jet(Scratch::Eth1B, Scratch::Eth1BT),
        delta_v,
        jet(Scratch::Eth2C, Scratch::Eth2CT),
        jet(Scratch::EthPrime2CSharp, Scratch::EthPrime2CSharpT)};
    constexpr Plus2ProductionJkDerivative metric_slots[7]{
        Plus2ProductionJkDerivative::CapitalDelta2CSharp,
        Plus2ProductionJkDerivative::EthPrime1BSharp,
        Plus2ProductionJkDerivative::CapitalDelta2C,
        Plus2ProductionJkDerivative::Eth1B,
        Plus2ProductionJkDerivative::CapitalDelta2V,
        Plus2ProductionJkDerivative::Eth2C,
        Plus2ProductionJkDerivative::EthPrime2CSharp};
    for (std::size_t i = 0; i < 7; ++i) {
      const std::size_t index = flat4(
          mode, static_cast<std::size_t>(metric_slots[i]), radial, theta, dc,
          radial_count, theta_count);
      jk_value[index] = metric_derivatives[i].value;
      jk_tangent[index] = metric_derivatives[i].dt;
    }
    put_s(Scratch::Delta2CSharp, delta_csharp.value);
    put_s(Scratch::Delta2CSharpT, delta_csharp.dt);
    put_s(Scratch::Delta2C, delta_c.value);
    put_s(Scratch::Delta2CT, delta_c.dt);
    put_s(Scratch::Delta2V, delta_v.value);
    put_s(Scratch::Delta2VT, delta_v.dt);
    put_s(Scratch::Delta2VSharp, delta_vsharp.value);
    put_s(Scratch::Delta2VSharpT, delta_vsharp.dt);
  }
};

struct RadialConnectionFunctor {
  const Complex* scratch_input;
  Complex* scratch_output;
  RadialDiscretization discretization;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t count = static_cast<std::size_t>(Scratch::Count);
    constexpr Scratch inputs[2]{Scratch::Sig, Scratch::Kap};
    constexpr Scratch outputs[2]{Scratch::DrSig, Scratch::DrKap};
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode_field = flat / plane;
    const std::size_t field = mode_field % 2;
    const std::size_t mode = mode_field / 2;
    const std::size_t within = flat - mode_field * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t base = flat4(
        mode, static_cast<std::size_t>(inputs[field]), 0, theta, count,
        radial_count, theta_count);
    scratch_output[flat4(mode, static_cast<std::size_t>(outputs[field]),
                         radial, theta, count, radial_count, theta_count)] =
        radial_first_derivative_strided_at(
            discretization, scratch_input + base, radial_count, radial,
            inverse_spacing, theta_count);
  }
};

struct FinalizeFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Complex* scratch;
  const std::uint8_t* ready;
  Complex* primitive_value;
  Complex* primitive_tangent;
  Complex* jk_value;
  Complex* jk_tangent;
  Complex* q_value;
  std::uint64_t* primitive_value_stamps;
  std::uint64_t* primitive_tangent_stamps;
  std::uint64_t* jk_value_stamps;
  std::uint64_t* jk_tangent_stamps;
  std::uint64_t* q_value_stamps;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t sc = static_cast<std::size_t>(Scratch::Count);
    constexpr std::size_t pc =
        static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
    constexpr std::size_t jc =
        static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
    constexpr std::size_t qc =
        static_cast<std::size_t>(Plus2ProductionQDerivative::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const auto s = [&](const Scratch field) {
      return scratch[flat4(mode, static_cast<std::size_t>(field), radial,
                           theta, sc, radial_count, theta_count)];
    };
    const bool valid = ready[0] != 0;
    const double radius = grid.coordinate(radial);
    const Complex delta_sig =
        valid ? delta_n_point(s(Scratch::Sig), s(Scratch::SigT),
                              s(Scratch::DrSig), 2, radius, parameters.mass,
                              parameters.compactification_length)
              : Complex{};
    const Complex delta_kap =
        valid ? delta_n_point(s(Scratch::Kap), s(Scratch::KapT),
                              s(Scratch::DrKap), 3, radius, parameters.mass,
                              parameters.compactification_length)
              : Complex{};
    const Complex q_values[3]{delta_sig, delta_kap,
                              valid ? s(Scratch::EthPrime3Kap) : Complex{}};
    for (std::size_t field = 0; field < qc; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, qc,
                                      radial_count, theta_count);
      q_value[index] = q_values[field];
      q_value_stamps[index] = valid ? generation : 0;
    }
    for (std::size_t field = 0; field < pc; ++field) {
      if (field == static_cast<std::size_t>(Plus2SpatialPrimitive::Z0) ||
          field == static_cast<std::size_t>(Plus2SpatialPrimitive::Z1)) {
        continue;
      }
      const std::size_t index = flat4(mode, field, radial, theta, pc,
                                      radial_count, theta_count);
      if (!valid) {
        primitive_value[index] = Complex{};
        primitive_tangent[index] = Complex{};
      }
      primitive_value_stamps[index] = valid ? generation : 0;
      primitive_tangent_stamps[index] = valid ? generation : 0;
    }
    constexpr Plus2ProductionJkDerivative owned_jk[7]{
        Plus2ProductionJkDerivative::CapitalDelta2CSharp,
        Plus2ProductionJkDerivative::EthPrime1BSharp,
        Plus2ProductionJkDerivative::CapitalDelta2C,
        Plus2ProductionJkDerivative::Eth1B,
        Plus2ProductionJkDerivative::CapitalDelta2V,
        Plus2ProductionJkDerivative::Eth2C,
        Plus2ProductionJkDerivative::EthPrime2CSharp};
    for (const Plus2ProductionJkDerivative component : owned_jk) {
      const std::size_t field = static_cast<std::size_t>(component);
      const std::size_t index = flat4(mode, field, radial, theta, jc,
                                      radial_count, theta_count);
      if (!valid) {
        jk_value[index] = Complex{};
        jk_tangent[index] = Complex{};
      }
      jk_value_stamps[index] = valid ? generation : 0;
      jk_tangent_stamps[index] = valid ? generation : 0;
    }
  }
};

static_assert(std::is_trivially_copyable_v<SetReadyFunctor>);
static_assert(std::is_trivially_copyable_v<ValidateFunctor>);
static_assert(std::is_trivially_copyable_v<PackFunctor>);
static_assert(std::is_trivially_copyable_v<RadialMetricFunctor>);
static_assert(std::is_trivially_copyable_v<MetricPointFunctor>);
static_assert(std::is_trivially_copyable_v<RadialConnectionFunctor>);
static_assert(std::is_trivially_copyable_v<FinalizeFunctor>);
static_assert(sizeof(MetricPointFunctor) < 1800);
static_assert(sizeof(FinalizeFunctor) < 1800);

}  // namespace plus2_primitive_spatial_detail

// Concrete preallocated producer for the source callback seam.  WriteTarget
// is intentionally structural so this standalone header need not depend on
// the live-composition owner; Plus2LiveSourceWriteTarget is the production
// target and tests may use an equivalent narrow target.
template <class ExecSpace = ExecutionSpace>
class Plus2SourcePrimitiveSpatialProducer {
 public:
  using execution_space = ExecSpace;
  using scratch_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
  using index_view = Kokkos::View<std::size_t*, MemorySpace>;
  using real_view = Kokkos::View<Real*, MemorySpace>;

  Plus2SourcePrimitiveSpatialProducer(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& grid, const KerrParameters& parameters,
      const int ell_max, const Plus2SpatialThetaView& cos_theta,
      const Plus2SpatialThetaView& sin_theta,
      const std::string& label = "plus2_source_primitive_spatial",
      const RadialDiscretization discretization =
          RadialDiscretization::D105)
      : registry_(registry),
        grid_(grid),
        parameters_(parameters),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        discretization_(discretization),
        sharp_(label + "_sharp", registry.size()),
        radius_(label + "_radius", grid.size()),
        mode_view_(label + "_modes", registry.size()),
        scratch_(label + "_scratch", registry.size(),
                 static_cast<std::size_t>(
                     plus2_primitive_spatial_detail::Scratch::Count),
                 grid.size(), sin_theta.extent(0)),
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
                     parameters) {
    if (!registry.is_closed_under_sharp() || registry.size() == 0 ||
        discretization != RadialDiscretization::D105 ||
        grid.size() < radial_minimum_points(discretization) || ell_max < 2 ||
        sin_theta.extent(0) == 0 ||
        sin_theta.extent(0) != cos_theta.extent(0) ||
        !std::isfinite(parameters.mass) ||
        !std::isfinite(parameters.spin) ||
        !std::isfinite(parameters.compactification_length) ||
        parameters.mass <= 0.0 ||
        std::abs(parameters.spin) > parameters.mass ||
        parameters.compactification_length <= 0.0) {
      throw std::invalid_argument(
          "spin +2 primitive producer requires a closed D10-5 geometry");
    }
    auto host_sharp = Kokkos::create_mirror_view(sharp_);
    auto host_radius = Kokkos::create_mirror_view(radius_);
    auto host_modes = Kokkos::create_mirror_view(mode_view_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_sharp(mode) = registry.sharp_index(registry.modes()[mode]);
      host_modes(mode) = registry.modes()[mode];
    }
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      host_radius(radial) = grid.coordinate(radial);
    }
    Kokkos::deep_copy(execution, sharp_, host_sharp);
    Kokkos::deep_copy(execution, radius_, host_radius);
    Kokkos::deep_copy(execution, mode_view_, host_modes);
  }

  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return discretization_;
  }
  [[nodiscard]] scratch_view scratch() const { return scratch_; }
  [[nodiscard]] Kokkos::View<const std::uint8_t*, MemorySpace> readiness()
      const {
    return ready_;
  }

  template <class WriteTarget>
  void evaluate(const execution_space& execution,
                const Plus2PrimitiveReconstructionStage& reconstruction,
                const Plus2TransportedCurvatureStage& curvature,
                const Plus2BianchiDerivativeStage& bianchi,
                const WriteTarget& target,
                const Plus2PrimitiveReconstructionOffsets offsets = {}) {
    validate(reconstruction, curvature, bianchi, target, offsets);
    using namespace plus2_primitive_spatial_detail;
    const std::size_t modes = registry_.size();
    const std::size_t radial = grid_.size();
    const std::size_t theta = sin_theta_.extent(0);
    const std::size_t points = modes * radial * theta;
    Kokkos::parallel_for(
        "plus2_primitive_set_ready",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        SetReadyFunctor{ready_.data()});
    Kokkos::parallel_for(
        "plus2_primitive_validate_inputs",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        ValidateFunctor{
            reconstruction.value.data(), reconstruction.tangent.data(),
            reconstruction.second_tangent.data(),
            reconstruction.value_stamps.data(),
            reconstruction.tangent_stamps.data(),
            reconstruction.second_tangent_stamps.data(),
            curvature.fields.data(), curvature.stamps.data(),
            bianchi.fields.data(), bianchi.stamps.data(), ready_.data(),
            target.generation, reconstruction.value.extent(1), radial, theta,
            offsets.H, offsets.Pi, offsets.B, offsets.C, offsets.U});
    Kokkos::parallel_for(
        "plus2_primitive_pack_reconstruction",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        PackFunctor{grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
                    reconstruction.value.data(),
                    reconstruction.tangent.data(),
                    reconstruction.second_tangent.data(), sharp_.data(),
                    ready_.data(), scratch_.data(),
                    reconstruction.value.extent(1), radial, theta, offsets.H,
                    offsets.Pi, offsets.B, offsets.C, offsets.U});
    Kokkos::parallel_for(
        "plus2_primitive_metric_radial",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             modes * 10 * radial * theta),
        RadialMetricFunctor{scratch_.data(), scratch_.data(), discretization_,
                            radial, theta, 1.0 / grid_.spacing()});
    apply_metric_angular(execution);
    Kokkos::parallel_for(
        "plus2_primitive_metric_points",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        MetricPointFunctor{
            grid_, parameters_, cos_theta_.data(), sin_theta_.data(),
            mode_view_.data(), scratch_.data(), scratch_.data(),
            target.primitive_value.data(), target.primitive_tangent.data(),
            target.jk_derivative_value.data(),
            target.jk_derivative_tangent.data(), radial, theta});
    Kokkos::parallel_for(
        "plus2_primitive_connection_radial",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             modes * 2 * radial * theta),
        RadialConnectionFunctor{scratch_.data(), scratch_.data(),
                                discretization_, radial, theta,
                                1.0 / grid_.spacing()});
    kap_angular_.ethprime(
        execution, scratch_, static_cast<std::size_t>(Scratch::Kap), scratch_,
        static_cast<std::size_t>(Scratch::KapT), radius_, sin_theta_,
        cos_theta_, scratch_,
        static_cast<std::size_t>(Scratch::EthPrime3Kap));
    Kokkos::parallel_for(
        "plus2_primitive_finalize",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        FinalizeFunctor{
            grid_, parameters_, scratch_.data(), ready_.data(),
            target.primitive_value.data(),
            target.primitive_tangent.data(),
            target.jk_derivative_value.data(),
            target.jk_derivative_tangent.data(),
            target.q_derivative_value.data(),
            target.primitive_value_stamps.data(),
            target.primitive_tangent_stamps.data(),
            target.jk_derivative_value_stamps.data(),
            target.jk_derivative_tangent_stamps.data(),
            target.q_derivative_value_stamps.data(), target.generation, radial,
            theta});
    last_generation_ = target.generation;
  }

  // Callable adapter for the live source producer seam.  The generic z_plus
  // arguments are intentionally absent: local curvature reconstruction and
  // endpoint quotients are not this producer's authority.
  template <class WriteTarget>
  void operator()(
      const execution_space& execution, const double,
      const Plus2PrimitiveReconstructionStage& reconstruction,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi, const WriteTarget& target,
      const Plus2PrimitiveReconstructionOffsets offsets = {}) {
    evaluate(execution, reconstruction, curvature, bianchi, target, offsets);
  }

 private:
  template <class View>
  bool rank4_shape(const View& view, const std::size_t fields) const {
    return view.extent(0) == registry_.size() &&
           view.extent(1) == fields && view.extent(2) == grid_.size() &&
           view.extent(3) == sin_theta_.extent(0);
  }

  template <class WriteTarget>
  void validate(const Plus2PrimitiveReconstructionStage& reconstruction,
                const Plus2TransportedCurvatureStage& curvature,
                const Plus2BianchiDerivativeStage& bianchi,
                const WriteTarget& target,
                const Plus2PrimitiveReconstructionOffsets& offsets) const {
    constexpr std::size_t primitive_count =
        static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
    constexpr std::size_t jk_count =
        static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
    constexpr std::size_t q_count =
        static_cast<std::size_t>(Plus2ProductionQDerivative::Count);
    constexpr std::size_t curvature_count = static_cast<std::size_t>(
        Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t bianchi_count = static_cast<std::size_t>(
        Plus2BianchiDerivativeComponent::Count);
    const std::size_t largest =
        std::max({offsets.H, offsets.Pi, offsets.B, offsets.C, offsets.U});
    const auto reconstruction_shape = [&](const auto& view) {
      return view.extent(0) == registry_.size() && view.extent(1) > largest &&
             view.extent(2) == grid_.size() &&
             view.extent(3) == sin_theta_.extent(0);
    };
    if (target.generation == 0 ||
        target.generation != reconstruction.generation ||
        target.generation <= last_generation_ ||
        !reconstruction_shape(reconstruction.value) ||
        !reconstruction_shape(reconstruction.tangent) ||
        !reconstruction_shape(reconstruction.second_tangent) ||
        !reconstruction_shape(reconstruction.value_stamps) ||
        !reconstruction_shape(reconstruction.tangent_stamps) ||
        !reconstruction_shape(reconstruction.second_tangent_stamps) ||
        !rank4_shape(curvature.fields, curvature_count) ||
        !rank4_shape(curvature.stamps, curvature_count) ||
        !rank4_shape(bianchi.fields, bianchi_count) ||
        !rank4_shape(bianchi.stamps, bianchi_count) ||
        !rank4_shape(target.primitive_value, primitive_count) ||
        !rank4_shape(target.primitive_tangent, primitive_count) ||
        !rank4_shape(target.jk_derivative_value, jk_count) ||
        !rank4_shape(target.jk_derivative_tangent, jk_count) ||
        !rank4_shape(target.q_derivative_value, q_count) ||
        !rank4_shape(target.primitive_value_stamps, primitive_count) ||
        !rank4_shape(target.primitive_tangent_stamps, primitive_count) ||
        !rank4_shape(target.jk_derivative_value_stamps, jk_count) ||
        !rank4_shape(target.jk_derivative_tangent_stamps, jk_count) ||
        !rank4_shape(target.q_derivative_value_stamps, q_count)) {
      throw std::invalid_argument(
          "spin +2 primitive producer stage contract is incomplete");
    }
  }

  void apply_metric_angular(const execution_space& execution) {
    using S = plus2_primitive_spatial_detail::Scratch;
    const auto eth = [&](auto& coordinator, const S value, const S tangent,
                         const S output) {
      coordinator.eth(execution, scratch_, static_cast<std::size_t>(value),
                      scratch_, static_cast<std::size_t>(tangent), radius_,
                      sin_theta_, cos_theta_, scratch_,
                      static_cast<std::size_t>(output));
    };
    const auto ethprime = [&](auto& coordinator, const S value,
                              const S tangent, const S output) {
      coordinator.ethprime(
          execution, scratch_, static_cast<std::size_t>(value), scratch_,
          static_cast<std::size_t>(tangent), radius_, sin_theta_, cos_theta_,
          scratch_, static_cast<std::size_t>(output));
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
  Plus2SpatialThetaView cos_theta_;
  Plus2SpatialThetaView sin_theta_;
  RadialDiscretization discretization_;
  index_view sharp_;
  real_view radius_;
  // Signed modes are stored separately because pointwise thorn contains m.
  Kokkos::View<int*, MemorySpace> mode_view_;
  scratch_view scratch_;
  Kokkos::View<std::uint8_t*, MemorySpace> ready_;
  SignedModeAngularCoordinator<execution_space> bsharp_angular_;
  SignedModeAngularCoordinator<execution_space> csharp_angular_;
  SignedModeAngularCoordinator<execution_space> v_angular_;
  SignedModeAngularCoordinator<execution_space> c_angular_;
  SignedModeAngularCoordinator<execution_space> b_angular_;
  SignedModeAngularCoordinator<execution_space> kap_angular_;
  std::uint64_t last_generation_ = 0;
};

}  // namespace teuk
