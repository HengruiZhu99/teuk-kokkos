#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_source_primitive_spatial.hpp"

namespace {

using C = teuk::Complex;
using execution_space = teuk::ExecutionSpace;

constexpr int ell = 3;
constexpr int theta_count = 8;
constexpr std::size_t input_fields = 5;
constexpr std::size_t pc =
    static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count);
constexpr std::size_t jc =
    static_cast<std::size_t>(teuk::Plus2ProductionJkDerivative::Count);
constexpr std::size_t qc =
    static_cast<std::size_t>(teuk::Plus2ProductionQDerivative::Count);
constexpr std::size_t cc = static_cast<std::size_t>(
    teuk::Plus2TransportedCurvatureComponent::Count);
constexpr std::size_t bc = static_cast<std::size_t>(
    teuk::Plus2BianchiDerivativeComponent::Count);

std::size_t p(const teuk::Plus2SpatialPrimitive x) {
  return static_cast<std::size_t>(x);
}
std::size_t j(const teuk::Plus2ProductionJkDerivative x) {
  return static_cast<std::size_t>(x);
}

int producer_allocations = 0;
int producer_fences = 0;

void count_producer_allocation(Kokkos::Tools::SpaceHandle, const char*,
                               const void*, std::uint64_t) {
  ++producer_allocations;
}
void count_producer_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++producer_fences;
}

struct OutputTarget {
  std::uint64_t generation;
  teuk::Plus2SpatialPrimitiveView primitive_value;
  teuk::Plus2SpatialPrimitiveView primitive_tangent;
  teuk::Plus2ProductionJkDerivativeView jk_derivative_value;
  teuk::Plus2ProductionJkDerivativeView jk_derivative_tangent;
  teuk::Plus2ProductionQDerivativeView q_derivative_value;
  teuk::Plus2LiveStampView primitive_value_stamps;
  teuk::Plus2LiveStampView primitive_tangent_stamps;
  teuk::Plus2LiveStampView jk_derivative_value_stamps;
  teuk::Plus2LiveStampView jk_derivative_tangent_stamps;
  teuk::Plus2LiveStampView q_derivative_value_stamps;
};

struct Fixture {
  execution_space execution;
  teuk::ModeRegistry registry;
  teuk::UniformRadialGrid grid;
  teuk::KerrParameters parameters{1.0, 0.0, 1.0};
  teuk::Plus2SpatialThetaView cos_theta;
  teuk::Plus2SpatialThetaView sin_theta;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> value;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> tangent;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> second;
  teuk::Plus2LiveStampView value_stamps;
  teuk::Plus2LiveStampView tangent_stamps;
  teuk::Plus2LiveStampView second_stamps;
  teuk::Plus2TransportedCurvatureStorageView curvature;
  teuk::Plus2LiveStampView curvature_stamps;
  teuk::Plus2BianchiDerivativeView bianchi;
  teuk::Plus2LiveStampView bianchi_stamps;
  teuk::Plus2SpatialPrimitiveView primitive_value;
  teuk::Plus2SpatialPrimitiveView primitive_tangent;
  teuk::Plus2ProductionJkDerivativeView jk_value;
  teuk::Plus2ProductionJkDerivativeView jk_tangent;
  teuk::Plus2ProductionQDerivativeView q_value;
  teuk::Plus2LiveStampView primitive_value_stamps;
  teuk::Plus2LiveStampView primitive_tangent_stamps;
  teuk::Plus2LiveStampView jk_value_stamps;
  teuk::Plus2LiveStampView jk_tangent_stamps;
  teuk::Plus2LiveStampView q_value_stamps;
  std::unique_ptr<teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>>
      producer;

  Fixture(const std::size_t radial_count = 25,
          std::vector<int> modes = {0})
      : registry(std::move(modes)),
        grid(radial_count, 0.0, 0.7),
        cos_theta("primitive_test_cos", theta_count),
        sin_theta("primitive_test_sin", theta_count),
        value("primitive_test_value", registry.size(), input_fields,
              radial_count, theta_count),
        tangent("primitive_test_tangent", registry.size(), input_fields,
                radial_count, theta_count),
        second("primitive_test_second", registry.size(), input_fields,
               radial_count, theta_count),
        value_stamps("primitive_test_value_stamps", registry.size(),
                     input_fields, radial_count, theta_count),
        tangent_stamps("primitive_test_tangent_stamps", registry.size(),
                       input_fields, radial_count, theta_count),
        second_stamps("primitive_test_second_stamps", registry.size(),
                      input_fields, radial_count, theta_count),
        curvature("primitive_test_curvature", registry.size(), cc,
                  radial_count, theta_count),
        curvature_stamps("primitive_test_curvature_stamps", registry.size(),
                         cc, radial_count, theta_count),
        bianchi("primitive_test_bianchi", registry.size(), bc, radial_count,
                theta_count),
        bianchi_stamps("primitive_test_bianchi_stamps", registry.size(), bc,
                       radial_count, theta_count),
        primitive_value("primitive_test_output", registry.size(), pc,
                        radial_count, theta_count),
        primitive_tangent("primitive_test_output_t", registry.size(), pc,
                          radial_count, theta_count),
        jk_value("primitive_test_jk", registry.size(), jc, radial_count,
                 theta_count),
        jk_tangent("primitive_test_jk_t", registry.size(), jc, radial_count,
                   theta_count),
        q_value("primitive_test_q", registry.size(), qc, radial_count,
                theta_count),
        primitive_value_stamps("primitive_test_output_stamps", registry.size(),
                               pc, radial_count, theta_count),
        primitive_tangent_stamps("primitive_test_output_t_stamps",
                                 registry.size(), pc, radial_count,
                                 theta_count),
        jk_value_stamps("primitive_test_jk_stamps", registry.size(), jc,
                        radial_count, theta_count),
        jk_tangent_stamps("primitive_test_jk_t_stamps", registry.size(), jc,
                          radial_count, theta_count),
        q_value_stamps("primitive_test_q_stamps", registry.size(), qc,
                       radial_count, theta_count) {
    const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
    auto host_cos = Kokkos::create_mirror_view(cos_theta);
    auto host_sin = Kokkos::create_mirror_view(sin_theta);
    for (int k = 0; k < theta_count; ++k) {
      host_cos(k) = angular_grid.x[k];
      host_sin(k) = std::sqrt(1.0 - angular_grid.x[k] * angular_grid.x[k]);
    }
    Kokkos::deep_copy(execution, cos_theta, host_cos);
    Kokkos::deep_copy(execution, sin_theta, host_sin);
    producer = std::make_unique<
        teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>>(
        execution, registry, grid, parameters, ell, cos_theta, sin_theta,
        "primitive_spatial_test", teuk::RadialDiscretization::D105);
    execution.fence("finish primitive fixture setup");
  }

  static int spin(const std::size_t field) {
    constexpr int spins[input_fields]{0, -1, -2, -1, 0};
    return spins[field];
  }

  static double radial_profile(const std::size_t field, const double r,
                               const bool exponential) {
    if (exponential) {
      const double rate = 0.35 + 0.04 * static_cast<double>(field);
      return std::exp(rate * r) * (1.0 + 0.07 * r);
    }
    const double f = static_cast<double>(field + 1);
    return 1.0 + 0.03 * f * r + 0.04 * r * r - 0.02 * r * r * r +
           0.01 * r * r * r * r;
  }

  static double radial_derivative(const std::size_t field, const double r,
                                  const bool exponential) {
    if (exponential) {
      const double rate = 0.35 + 0.04 * static_cast<double>(field);
      return std::exp(rate * r) *
             (rate * (1.0 + 0.07 * r) + 0.07);
    }
    const double f = static_cast<double>(field + 1);
    return 0.03 * f + 0.08 * r - 0.06 * r * r + 0.04 * r * r * r;
  }

  void fill(const std::uint64_t generation, const double time = 0.13,
            const double amplitude = 1.0,
            const bool exponential = false) {
    auto hv = Kokkos::create_mirror_view(value);
    auto ht = Kokkos::create_mirror_view(tangent);
    auto hs = Kokkos::create_mirror_view(second);
    const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      const int m = registry.modes()[mode];
      const C mode_amplitude =
          amplitude * C(1.0 + 0.2 * static_cast<double>(m),
                        0.11 * static_cast<double>(m));
      for (std::size_t field = 0; field < input_fields; ++field) {
        const double rate = 0.12 + 0.015 * static_cast<double>(field);
        const double acceleration = -0.025 + 0.003 * field;
        const double tf = 1.0 + rate * time +
                          0.5 * acceleration * time * time;
        const double ttf = rate + acceleration * time;
        for (std::size_t radial = 0; radial < grid.size(); ++radial) {
          const double profile =
              radial_profile(field, grid.coordinate(radial), exponential);
          for (int theta = 0; theta < theta_count; ++theta) {
            const double harmonic = teuk::angular::spin_weighted_harmonic_theta(
                ell, m, spin(field), angular_grid.theta(theta));
            const C base = mode_amplitude * profile * harmonic;
            hv(mode, field, radial, theta) = tf * base;
            ht(mode, field, radial, theta) = ttf * base;
            hs(mode, field, radial, theta) = acceleration * base;
          }
        }
      }
    }
    Kokkos::deep_copy(execution, value, hv);
    Kokkos::deep_copy(execution, tangent, ht);
    Kokkos::deep_copy(execution, second, hs);
    Kokkos::deep_copy(execution, value_stamps, generation);
    Kokkos::deep_copy(execution, tangent_stamps, generation);
    Kokkos::deep_copy(execution, second_stamps, generation);
    Kokkos::deep_copy(execution, curvature, C(0.17, -0.06));
    Kokkos::deep_copy(execution, curvature_stamps, generation);
    Kokkos::deep_copy(execution, bianchi, C(-0.09, 0.04));
    Kokkos::deep_copy(execution, bianchi_stamps, generation);
  }

  OutputTarget target(const std::uint64_t generation) const {
    return {generation,
            primitive_value,
            primitive_tangent,
            jk_value,
            jk_tangent,
            q_value,
            primitive_value_stamps,
            primitive_tangent_stamps,
            jk_value_stamps,
            jk_tangent_stamps,
            q_value_stamps};
  }

  void evaluate(const std::uint64_t generation) {
    const teuk::Plus2PrimitiveReconstructionStage stage{
        generation, value, tangent, second, value_stamps, tangent_stamps,
        second_stamps};
    producer->evaluate(execution, stage, {curvature, curvature_stamps},
                       {bianchi, bianchi_stamps}, target(generation));
  }
};

template <class Scalar>
std::array<Scalar, 12> oracle_fields(
    const double radius, const teuk::Plus2PrimitiveBackground& background,
    const teuk::plus2_primitive_spatial_detail::MetricFieldsT<Scalar>& f,
    const teuk::plus2_primitive_spatial_detail::MetricDerivativesT<Scalar>& d) {
  teuk::Plus2ReconstructionPrimitiveInputsT<Scalar> fields{
      f.U, f.Usharp, f.C, f.Csharp, f.B, f.Bsharp, f.H, f.Pi};
  teuk::Plus2PrimitiveDerivativesT<Scalar> derivatives{};
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
  const auto result =
      teuk::plus2_source_primitives(radius, background, fields, derivatives);
  return {result.H,  result.Sig, result.Kap, result.Rh,
          result.Ta, result.Al,  result.Be,  result.Ep,
          result.Pi, result.V,   result.C,   result.B};
}

TEST_CASE("plus2 primitive point formulas and Jet tangents match reviewed oracle") {
  using J = teuk::Jet1<C>;
  const teuk::KerrParameters parameters{1.0, 0.73, 1.4};
  const double r = 0.31;
  const double x = -0.27;
  const double sin = std::sqrt(1.0 - x * x);
  const auto background =
      teuk::plus2_primitive_background(parameters, r, x, sin);
  const auto z = [](const double a, const double b, const double da,
                    const double db) { return J{{a, b}, {da, db}}; };
  const teuk::plus2_primitive_spatial_detail::MetricFieldsT<J> fields{
      z(.3, -.2, .04, .01), z(-.1, .25, .02, -.03),
      z(.12, .07, -.01, .05), z(.2, -.08, .03, .02),
      z(-.17, .11, .01, -.04), z(.09, .13, -.02, .01),
      z(.21, -.05, .06, -.02), z(-.14, .19, .02, .03)};
  const teuk::plus2_primitive_spatial_detail::MetricDerivativesT<J> d{
      z(.03, -.02, .01, .004), z(-.04, .06, .007, -.008),
      z(.08, .01, -.01, .002), z(-.03, .04, .006, .005),
      z(.02, -.07, -.003, .009), z(.05, .02, .004, -.006),
      z(-.06, .03, .008, .001), z(.01, .09, -.002, .007),
      z(.07, -.04, .005, -.003), z(-.02, .05, .003, .004),
      z(.04, .08, -.007, .002)};
  const auto actual =
      teuk::plus2_primitive_spatial_detail::connection_primitives(
          r, background, fields, d);
  const std::array<J, 12> actual_fields{
      actual.H,  actual.Sig, actual.Kap, actual.Rh,
      actual.Ta, actual.Al,  actual.Be,  actual.Ep,
      actual.Pi, actual.V,   actual.C,   actual.B};
  const auto expected = oracle_fields(r, background, fields, d);
  for (std::size_t i = 0; i < actual_fields.size(); ++i) {
    CHECK_COMPLEX_NEAR(actual_fields[i].value, expected[i].value, 2.0e-14);
    CHECK_COMPLEX_NEAR(actual_fields[i].dt, expected[i].dt, 2.0e-14);
  }
}

TEST_CASE("plus2 primitive producer owns only twelve primitives seven JK and three Q slots") {
  Fixture fixture;
  constexpr std::uint64_t generation = 7;
  fixture.fill(generation);
  const C sentinel(8.25, -3.5);
  constexpr std::uint64_t stamp_sentinel = 991;
  Kokkos::deep_copy(fixture.execution, fixture.primitive_value, sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.primitive_tangent, sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.jk_value, sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.jk_tangent, sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.q_value, sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.primitive_value_stamps,
                    stamp_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.primitive_tangent_stamps,
                    stamp_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.jk_value_stamps,
                    stamp_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.jk_tangent_stamps,
                    stamp_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.q_value_stamps,
                    stamp_sentinel);
  fixture.evaluate(generation);
  fixture.execution.fence("finish primitive ownership test");
  const auto pv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       fixture.primitive_value);
  const auto ps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primitive_value_stamps);
  const auto jv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       fixture.jk_value);
  const auto js = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.jk_value_stamps);
  const auto qs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.q_value_stamps);
  for (const auto slot : {teuk::Plus2SpatialPrimitive::Z0,
                          teuk::Plus2SpatialPrimitive::Z1}) {
    CHECK_COMPLEX_NEAR(pv(0, p(slot), 0, 0), sentinel, 0.0);
    CHECK(ps(0, p(slot), 0, 0) == stamp_sentinel);
  }
  for (std::size_t field = 2; field < pc; ++field) {
    CHECK(ps(0, field, 0, 0) == generation);
    CHECK(Kokkos::abs(pv(0, field, 0, 0) - sentinel) > 1.0e-5);
  }
  for (const auto slot : {teuk::Plus2ProductionJkDerivative::CapitalDelta4Z1,
                          teuk::Plus2ProductionJkDerivative::EthPrime4Z1,
                          teuk::Plus2ProductionJkDerivative::CapitalDelta5Z0,
                          teuk::Plus2ProductionJkDerivative::Eth5Z0}) {
    CHECK_COMPLEX_NEAR(jv(0, j(slot), 0, 0), sentinel, 0.0);
    CHECK(js(0, j(slot), 0, 0) == stamp_sentinel);
  }
  for (const auto slot : {
           teuk::Plus2ProductionJkDerivative::CapitalDelta2CSharp,
           teuk::Plus2ProductionJkDerivative::EthPrime1BSharp,
           teuk::Plus2ProductionJkDerivative::CapitalDelta2C,
           teuk::Plus2ProductionJkDerivative::Eth1B,
           teuk::Plus2ProductionJkDerivative::CapitalDelta2V,
           teuk::Plus2ProductionJkDerivative::Eth2C,
           teuk::Plus2ProductionJkDerivative::EthPrime2CSharp}) {
    CHECK(js(0, j(slot), 0, 0) == generation);
    CHECK(Kokkos::abs(jv(0, j(slot), 0, 0) - sentinel) > 1.0e-5);
  }
  for (std::size_t field = 0; field < qc; ++field) {
    CHECK(qs(0, field, 0, 0) == generation);
  }

  // Independent host point oracle: consume the spatial graph's explicit
  // operator slots, then call the pre-existing reviewed primitive evaluator.
  using S = teuk::plus2_primitive_spatial_detail::Scratch;
  using J = teuk::Jet1<C>;
  const auto scratch = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.producer->scratch());
  const std::size_t radial = fixture.grid.size() / 2;
  const std::size_t theta = 3;
  const auto s = [&](const S field) {
    return scratch(0, static_cast<std::size_t>(field), radial, theta);
  };
  const auto jet = [&](const S value_slot, const S tangent_slot) {
    return J{s(value_slot), s(tangent_slot)};
  };
  const teuk::plus2_primitive_spatial_detail::MetricFieldsT<J> fields{
      jet(S::U, S::UT),           jet(S::USharp, S::USharpT),
      jet(S::C, S::CT),           jet(S::CSharp, S::CSharpT),
      jet(S::B, S::BT),           jet(S::BSharp, S::BSharpT),
      jet(S::H, S::HT),           jet(S::Pi, S::PiT)};
  const teuk::plus2_primitive_spatial_detail::MetricDerivativesT<J>
      derivatives{
          jet(S::Thorn1BSharp, S::Thorn1BSharpT),
          jet(S::Thorn2CSharp, S::Thorn2CSharpT),
          jet(S::Eth2V, S::Eth2VT),
          jet(S::EthPrime2CSharp, S::EthPrime2CSharpT),
          jet(S::Eth2C, S::Eth2CT),
          jet(S::Delta2CSharp, S::Delta2CSharpT),
          jet(S::Delta2C, S::Delta2CT),
          jet(S::Delta2V, S::Delta2VT),
          jet(S::Delta2VSharp, S::Delta2VSharpT),
          jet(S::Eth1B, S::Eth1BT),
          jet(S::EthPrime1BSharp, S::EthPrime1BSharpT)};
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  const double radius_value = fixture.grid.coordinate(radial);
  const auto background = teuk::plus2_primitive_background(
      fixture.parameters, radius_value, angular_grid.x[theta],
      std::sqrt(1.0 - angular_grid.x[theta] * angular_grid.x[theta]));
  const auto expected =
      oracle_fields(radius_value, background, fields, derivatives);
  constexpr teuk::Plus2SpatialPrimitive slots[12]{
      teuk::Plus2SpatialPrimitive::H,  teuk::Plus2SpatialPrimitive::Sig,
      teuk::Plus2SpatialPrimitive::Kap, teuk::Plus2SpatialPrimitive::Rh,
      teuk::Plus2SpatialPrimitive::Ta, teuk::Plus2SpatialPrimitive::Al,
      teuk::Plus2SpatialPrimitive::Be, teuk::Plus2SpatialPrimitive::Ep,
      teuk::Plus2SpatialPrimitive::Pi, teuk::Plus2SpatialPrimitive::V,
      teuk::Plus2SpatialPrimitive::C,  teuk::Plus2SpatialPrimitive::B};
  const auto pt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primitive_tangent);
  for (std::size_t i = 0; i < 12; ++i) {
    CHECK_COMPLEX_NEAR(pv(0, p(slots[i]), radial, theta), expected[i].value,
                       3.0e-13);
    CHECK_COMPLEX_NEAR(pt(0, p(slots[i]), radial, theta), expected[i].dt,
                       3.0e-13);
  }
}

TEST_CASE("plus2 primitive producer fails globally closed on one stale stage input") {
  Fixture fixture;
  constexpr std::uint64_t generation = 13;
  fixture.fill(generation);
  auto host_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.second_stamps);
  host_stamps(0, 4, fixture.grid.size() / 2, 2) = generation - 1;
  Kokkos::deep_copy(fixture.execution, fixture.second_stamps, host_stamps);
  const C transport_sentinel(-7.0, 2.5);
  constexpr std::uint64_t transport_stamp = 812;
  Kokkos::deep_copy(fixture.execution, fixture.primitive_value,
                    transport_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.jk_value, transport_sentinel);
  Kokkos::deep_copy(fixture.execution, fixture.primitive_value_stamps,
                    transport_stamp);
  Kokkos::deep_copy(fixture.execution, fixture.jk_value_stamps,
                    transport_stamp);
  fixture.evaluate(generation);
  fixture.execution.fence("finish primitive stale-input test");
  const auto ready = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.producer->readiness());
  const auto pv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       fixture.primitive_value);
  const auto ps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primitive_value_stamps);
  const auto jv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       fixture.jk_value);
  const auto js = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.jk_value_stamps);
  CHECK(ready(0) == 0);
  CHECK_COMPLEX_NEAR(pv(0, p(teuk::Plus2SpatialPrimitive::Sig), 0, 0), C{},
                     0.0);
  CHECK(ps(0, p(teuk::Plus2SpatialPrimitive::Sig), 0, 0) == 0);
  CHECK_COMPLEX_NEAR(jv(0, j(teuk::Plus2ProductionJkDerivative::Eth1B), 0, 0),
                     C{}, 0.0);
  CHECK(js(0, j(teuk::Plus2ProductionJkDerivative::Eth1B), 0, 0) == 0);
  CHECK_COMPLEX_NEAR(pv(0, p(teuk::Plus2SpatialPrimitive::Z0), 0, 0),
                     transport_sentinel, 0.0);
  CHECK(ps(0, p(teuk::Plus2SpatialPrimitive::Z0), 0, 0) == transport_stamp);
  CHECK_COMPLEX_NEAR(
      jv(0, j(teuk::Plus2ProductionJkDerivative::CapitalDelta4Z1), 0, 0),
      transport_sentinel, 0.0);
  CHECK(js(0, j(teuk::Plus2ProductionJkDerivative::CapitalDelta4Z1), 0, 0) ==
        transport_stamp);
}

TEST_CASE("plus2 primitive producer fails closed on nonfinite h2 input") {
  Fixture fixture;
  constexpr std::uint64_t generation = 21;
  fixture.fill(generation);
  auto host_second = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.second);
  host_second(0, 4, fixture.grid.size() / 2, 2) =
      C(std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(fixture.execution, fixture.second, host_second);
  fixture.evaluate(generation);
  fixture.execution.fence("finish primitive nonfinite test");
  const auto ready = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.producer->readiness());
  const auto stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.q_value_stamps);
  CHECK(ready(0) == 0);
  CHECK(stamps(0, 0, 0, 0) == 0);
}

TEST_CASE("plus2 primitive signed sharp reads only the negative mode partner") {
  auto run = [](const double negative_scale, const double positive_scale) {
    Fixture fixture(25, {-1, 1});
    fixture.fill(1);
    auto scale_mode = [&](const std::size_t mode, const double scale) {
      for (auto* view : {&fixture.value, &fixture.tangent, &fixture.second}) {
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        *view);
        for (const std::size_t field : {std::size_t{2}, std::size_t{3}}) {
          for (std::size_t radial = 0; radial < fixture.grid.size(); ++radial) {
            for (int theta = 0; theta < theta_count; ++theta) {
              host(mode, field, radial, theta) *= scale;
            }
          }
        }
        Kokkos::deep_copy(fixture.execution, *view, host);
      }
    };
    scale_mode(0, negative_scale);
    scale_mode(1, positive_scale);
    fixture.evaluate(1);
    fixture.execution.fence("finish primitive signed sharp sample");
    const auto output = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.primitive_value);
    return output(1, p(teuk::Plus2SpatialPrimitive::Sig),
                  fixture.grid.size() / 2, 3);
  };
  const C base = run(1.0, 1.0);
  const C positive_changed = run(1.0, 1.7);
  const C negative_changed = run(1.7, 1.0);
  CHECK_COMPLEX_NEAR(base, positive_changed, 2.0e-13);
  CHECK(Kokkos::abs(base - negative_changed) > 1.0e-5);
}

TEST_CASE("plus2 primitive values and Jet tangents scale linearly") {
  Fixture base;
  Fixture scaled;
  constexpr double factor = 1.7;
  base.fill(1, 0.13, 1.0);
  scaled.fill(1, 0.13, factor);
  base.evaluate(1);
  scaled.evaluate(1);
  base.execution.fence("finish primitive amplitude test");
  const auto bv = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, base.primitive_value);
  const auto bt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, base.primitive_tangent);
  const auto sv = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, scaled.primitive_value);
  const auto st = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, scaled.primitive_tangent);
  const auto bq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       base.q_value);
  const auto sq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       scaled.q_value);
  const std::size_t radial = base.grid.size() / 2;
  constexpr std::size_t theta = 3;
  for (std::size_t field = 2; field < pc; ++field) {
    CHECK_COMPLEX_NEAR(sv(0, field, radial, theta),
                       factor * bv(0, field, radial, theta), 4.0e-13);
    CHECK_COMPLEX_NEAR(st(0, field, radial, theta),
                       factor * bt(0, field, radial, theta), 4.0e-13);
  }
  for (std::size_t field = 0; field < qc; ++field) {
    CHECK_COMPLEX_NEAR(sq(0, field, radial, theta),
                       factor * bq(0, field, radial, theta), 4.0e-13);
  }
}

double endpoint_sigma_error(const std::size_t points) {
  Fixture fixture(points);
  constexpr std::uint64_t generation = 1;
  fixture.fill(generation, 0.13, 1.0, true);
  fixture.evaluate(generation);
  fixture.execution.fence("finish primitive convergence sample");
  const auto produced = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primitive_value);
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  double error2 = 0.0;
  for (const std::size_t radial : {std::size_t{0}, points - 1}) {
    const double r = fixture.grid.coordinate(radial);
    for (int theta = 0; theta < theta_count; ++theta) {
      const double angle = angular_grid.theta(theta);
      const double y2 =
          teuk::angular::spin_weighted_harmonic_theta(ell, 0, 2, angle);
      const double y1 =
          teuk::angular::spin_weighted_harmonic_theta(ell, 0, 1, angle);
      const std::size_t B = 2;
      const std::size_t C_field = 3;
      const double tb = 1.0 + (0.12 + 0.015 * B) * 0.13 +
                        0.5 * (-0.025 + 0.003 * B) * 0.13 * 0.13;
      const double ttb = (0.12 + 0.015 * B) +
                         (-0.025 + 0.003 * B) * 0.13;
      const double tc = 1.0 + (0.12 + 0.015 * C_field) * 0.13 +
                        0.5 * (-0.025 + 0.003 * C_field) * 0.13 * 0.13;
      const C b = tb * Fixture::radial_profile(B, r, true) * y2;
      const C bt = ttb * Fixture::radial_profile(B, r, true) * y2;
      const C br = tb * Fixture::radial_derivative(B, r, true) * y2;
      const C c = tc * Fixture::radial_profile(C_field, r, true) * y1;
      const auto background = teuk::plus2_primitive_background(
          fixture.parameters, r, angular_grid.x[theta],
          std::sqrt(1.0 - angular_grid.x[theta] * angular_grid.x[theta]));
      const C thorn_b = teuk::thorn_n_point(
          b, bt, br, 1, 2, 0, 0, r, angular_grid.x[theta],
          fixture.parameters.mass, fixture.parameters.spin,
          fixture.parameters.compactification_length,
          background.kerr.epsilon0);
      const C expected =
          0.5 * thorn_b +
          0.5 * (background.kerr.rho0 -
                 Kokkos::conj(background.kerr.rho0)) *
              b -
          r * r * (Kokkos::conj(background.kerr.pi0) +
                   background.kerr.tau0) *
              c;
      const C difference =
          produced(0, p(teuk::Plus2SpatialPrimitive::Sig), radial, theta) -
          expected;
      error2 += Kokkos::abs(difference) * Kokkos::abs(difference);
    }
  }
  return std::sqrt(error2 / (2.0 * theta_count));
}

C analytic_kap(const Fixture& fixture, const double r, const int theta) {
  constexpr std::size_t C_field = 3;
  constexpr std::size_t U_field = 4;
  constexpr double time = 0.13;
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  const double angle = angular_grid.theta(theta);
  const double x = angular_grid.x[theta];
  const double sin = std::sqrt(1.0 - x * x);
  const auto background =
      teuk::plus2_primitive_background(fixture.parameters, r, x, sin);
  const auto time_factors = [](const std::size_t field) {
    const double rate = 0.12 + 0.015 * static_cast<double>(field);
    const double acceleration = -0.025 + 0.003 * field;
    return std::array<double, 3>{
        1.0 + rate * time + 0.5 * acceleration * time * time,
        rate + acceleration * time, acceleration};
  };
  const auto ct = time_factors(C_field);
  const auto ut = time_factors(U_field);
  const double yc = teuk::angular::spin_weighted_harmonic_theta(
      ell, 1, 1, angle);
  const double yu = teuk::angular::spin_weighted_harmonic_theta(
      ell, 1, Fixture::spin(U_field), angle);
  const double yu_raised = teuk::angular::raising_factor(ell, 0) *
                           teuk::angular::spin_weighted_harmonic_theta(
                               ell, 1, 1, angle);
  const C csharp_amplitude(0.8, 0.11);
  const C u_amplitude(1.2, 0.11);
  const C csharp =
      csharp_amplitude * ct[0] *
      Fixture::radial_profile(C_field, r, true) * yc;
  const C csharp_t =
      csharp_amplitude * ct[1] *
      Fixture::radial_profile(C_field, r, true) * yc;
  const C csharp_r =
      csharp_amplitude * ct[0] *
      Fixture::radial_derivative(C_field, r, true) * yc;
  const C u = u_amplitude * ut[0] *
              Fixture::radial_profile(U_field, r, true) * yu;
  const C u_t = u_amplitude * ut[1] *
                Fixture::radial_profile(U_field, r, true) * yu;
  // For a=0, mu0=-1 and hence V=-U, including its stage tangent.
  const C v = -u;
  const C v_t = -u_t;
  const C v_raised =
      -u_amplitude * ut[0] *
      Fixture::radial_profile(U_field, r, true) * yu_raised;
  const C thorn_c = teuk::thorn_n_point(
      csharp, csharp_t, csharp_r, 2, 1, 1, 1, r, x,
      fixture.parameters.mass, fixture.parameters.spin,
      fixture.parameters.compactification_length,
      background.kerr.epsilon0);
  const C eth_v = teuk::eth_n_point(
      v, v_t, v_raised, 0, 2, r, sin, x, fixture.parameters.spin,
      fixture.parameters.compactification_length);
  return thorn_c - Kokkos::conj(background.kerr.rho0) * csharp -
         0.5 * eth_v -
         0.5 * r * (Kokkos::conj(background.kerr.pi0) +
                    background.kerr.tau0) *
             v;
}

C analytic_kap_t(const Fixture& fixture, const double r, const int theta) {
  constexpr std::size_t C_field = 3;
  constexpr std::size_t U_field = 4;
  constexpr double time = 0.13;
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  const double angle = angular_grid.theta(theta);
  const double x = angular_grid.x[theta];
  const double sin = std::sqrt(1.0 - x * x);
  const auto background =
      teuk::plus2_primitive_background(fixture.parameters, r, x, sin);
  const auto factors = [](const std::size_t field) {
    const double rate = 0.12 + 0.015 * static_cast<double>(field);
    const double acceleration = -0.025 + 0.003 * field;
    return std::array<double, 2>{rate + acceleration * time, acceleration};
  };
  const auto ct = factors(C_field);
  const auto ut = factors(U_field);
  const double yc = teuk::angular::spin_weighted_harmonic_theta(
      ell, 1, 1, angle);
  const double yu = teuk::angular::spin_weighted_harmonic_theta(
      ell, 1, Fixture::spin(U_field), angle);
  const double yu_raised = teuk::angular::raising_factor(ell, 0) *
                           teuk::angular::spin_weighted_harmonic_theta(
                               ell, 1, 1, angle);
  const C csharp_amplitude(0.8, 0.11);
  const C u_amplitude(1.2, 0.11);
  const C c_t = csharp_amplitude * ct[0] *
                Fixture::radial_profile(C_field, r, true) * yc;
  const C c_tt = csharp_amplitude * ct[1] *
                 Fixture::radial_profile(C_field, r, true) * yc;
  const C c_tr =
      csharp_amplitude * ct[0] *
      Fixture::radial_derivative(C_field, r, true) * yc;
  const C v_t =
      -u_amplitude * ut[0] *
      Fixture::radial_profile(U_field, r, true) * yu;
  const C v_tt =
      -u_amplitude * ut[1] *
      Fixture::radial_profile(U_field, r, true) * yu;
  const C v_t_raised =
      -u_amplitude * ut[0] *
      Fixture::radial_profile(U_field, r, true) * yu_raised;
  const C thorn_c_t = teuk::thorn_n_point(
      c_t, c_tt, c_tr, 2, 1, 1, 1, r, x, fixture.parameters.mass,
      fixture.parameters.spin, fixture.parameters.compactification_length,
      background.kerr.epsilon0);
  const C eth_v_t = teuk::eth_n_point(
      v_t, v_tt, v_t_raised, 0, 2, r, sin, x, fixture.parameters.spin,
      fixture.parameters.compactification_length);
  return thorn_c_t - Kokkos::conj(background.kerr.rho0) * c_t -
         0.5 * eth_v_t -
         0.5 * r * (Kokkos::conj(background.kerr.pi0) +
                    background.kerr.tau0) *
             v_t;
}

C eighth_order_derivative(const Fixture& fixture, const double r,
                           const int theta) {
  constexpr double h = 2.0e-3;
  return (3.0 * analytic_kap(fixture, r - 4.0 * h, theta) -
          32.0 * analytic_kap(fixture, r - 3.0 * h, theta) +
          168.0 * analytic_kap(fixture, r - 2.0 * h, theta) -
          672.0 * analytic_kap(fixture, r - h, theta) +
          672.0 * analytic_kap(fixture, r + h, theta) -
          168.0 * analytic_kap(fixture, r + 2.0 * h, theta) +
          32.0 * analytic_kap(fixture, r + 3.0 * h, theta) -
          3.0 * analytic_kap(fixture, r + 4.0 * h, theta)) /
         (840.0 * h);
}

double endpoint_delta3_kap_error(const std::size_t points) {
  Fixture fixture(points, {-1, 1});
  fixture.fill(1, 0.13, 1.0, true);
  fixture.evaluate(1);
  fixture.execution.fence("finish primitive Q convergence sample");
  const auto produced = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.q_value);
  double error2 = 0.0;
  for (const std::size_t radial : {std::size_t{0}, points - 1}) {
    const double r = fixture.grid.coordinate(radial);
    for (int theta = 0; theta < theta_count; ++theta) {
      const C expected = teuk::delta_n_point(
          analytic_kap(fixture, r, theta),
          analytic_kap_t(fixture, r, theta),
          eighth_order_derivative(fixture, r, theta), 3, r,
          fixture.parameters.mass, fixture.parameters.compactification_length);
      const C difference =
          produced(1, static_cast<std::size_t>(
                          teuk::Plus2ProductionQDerivative::CapitalDelta3Kap),
                   radial, theta) -
          expected;
      error2 += Kokkos::abs(difference) * Kokkos::abs(difference);
    }
  }
  return std::sqrt(error2 / (2.0 * theta_count));
}

TEST_CASE("plus2 primitive D10-5 graph converges at both radial endpoints") {
  const double coarse = endpoint_sigma_error(25);
  const double medium = endpoint_sigma_error(49);
  const double fine = endpoint_sigma_error(97);
  CHECK(coarse / medium > 24.0);
  CHECK(medium / fine > 24.0);
}

TEST_CASE("plus2 primitive deepest Q radial slot is endpoint fourth order") {
  const double coarse = endpoint_delta3_kap_error(25);
  const double medium = endpoint_delta3_kap_error(49);
  const double fine = endpoint_delta3_kap_error(97);
  CHECK(coarse / medium > 15.0);
  CHECK(medium / fine > 15.0);
}

TEST_CASE("plus2 primitive hot producer allocates and fences nothing") {
  Fixture fixture;
  fixture.fill(1);
  fixture.evaluate(1);
  fixture.execution.fence("warm primitive producer");
  fixture.fill(2);
  producer_allocations = 0;
  producer_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_producer_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_producer_fence);
  fixture.evaluate(2);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish primitive hot-path audit");
  CHECK(producer_allocations == 0);
  CHECK(producer_fences == 0);
}

}  // namespace
