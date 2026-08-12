#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "teuk/plus2_source_value_spatial.hpp"

namespace {

using C = teuk::Complex;
using J = teuk::Jet1<C>;
using Host4 =
    Kokkos::View<C****, Kokkos::LayoutRight, Kokkos::HostSpace>;

constexpr std::size_t p(const teuk::Plus2SpatialPrimitive x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t jkd(const teuk::Plus2ProductionJkDerivative x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t qd(const teuk::Plus2ProductionQDerivative x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t fam(const teuk::Plus2SpatialPairFamily x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t agg(const teuk::Plus2SpatialAggregate x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t jkagg(const teuk::Plus2ProductionJkAggregate x) {
  return static_cast<std::size_t>(x);
}
constexpr std::size_t outer(const teuk::Plus2SpatialOuterDerivative x) {
  return static_cast<std::size_t>(x);
}

struct Inputs {
  teuk::Plus2SpatialPrimitiveView primitive;
  teuk::Plus2SpatialPrimitiveView primitive_tangent;
  teuk::Plus2ProductionJkDerivativeView jk;
  teuk::Plus2ProductionJkDerivativeView jk_tangent;
  teuk::Plus2ProductionQDerivativeView q;
  Host4 hp, hpt, hjk, hjkt, hq;

  Inputs(const std::size_t modes, const std::size_t radial,
         const std::size_t theta, const char* label)
      : primitive(label, modes,
                  static_cast<std::size_t>(
                      teuk::Plus2SpatialPrimitive::Count),
                  radial, theta),
        primitive_tangent("plus2_value_primitive_t", modes,
                          static_cast<std::size_t>(
                              teuk::Plus2SpatialPrimitive::Count),
                          radial, theta),
        jk("plus2_value_jk", modes,
           static_cast<std::size_t>(
               teuk::Plus2ProductionJkDerivative::Count),
           radial, theta),
        jk_tangent("plus2_value_jk_t", modes,
                   static_cast<std::size_t>(
                       teuk::Plus2ProductionJkDerivative::Count),
                   radial, theta),
        q("plus2_value_q", modes,
          static_cast<std::size_t>(
              teuk::Plus2ProductionQDerivative::Count),
          radial, theta),
        hp(Kokkos::create_mirror_view(primitive)),
        hpt(Kokkos::create_mirror_view(primitive_tangent)),
        hjk(Kokkos::create_mirror_view(jk)),
        hjkt(Kokkos::create_mirror_view(jk_tangent)),
        hq(Kokkos::create_mirror_view(q)) {}

  void fill(const double amplitude = 1.0) {
    for (std::size_t mode = 0; mode < primitive.extent(0); ++mode) {
      for (std::size_t radial = 0; radial < primitive.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < primitive.extent(3); ++theta) {
          for (std::size_t component = 0; component < primitive.extent(1);
               ++component) {
            const double tag = 0.19 + 0.17 * mode + 0.021 * radial -
                               0.014 * theta + 0.009 * component;
            hp(mode, component, radial, theta) =
                amplitude * C(tag, -0.37 * tag + 0.01 * component);
            hpt(mode, component, radial, theta) =
                amplitude * C(-0.11 * tag, 0.23 * tag);
          }
          for (std::size_t component = 0; component < jk.extent(1);
               ++component) {
            const double tag = -0.13 + 0.08 * mode + 0.017 * radial +
                               0.012 * theta - 0.006 * component;
            hjk(mode, component, radial, theta) =
                amplitude * C(tag, 0.31 * tag);
            hjkt(mode, component, radial, theta) =
                amplitude * C(0.16 * tag, -0.09 * tag);
          }
          for (std::size_t component = 0; component < q.extent(1);
               ++component) {
            const double tag = 0.07 - 0.05 * mode + 0.01 * radial -
                               0.015 * theta + 0.02 * component;
            hq(mode, component, radial, theta) =
                amplitude * C(tag, -0.27 * tag);
          }
        }
      }
    }
    Kokkos::deep_copy(primitive, hp);
    Kokkos::deep_copy(primitive_tangent, hpt);
    Kokkos::deep_copy(jk, hjk);
    Kokkos::deep_copy(jk_tangent, hjkt);
    Kokkos::deep_copy(q, hq);
  }
};

struct Angles {
  teuk::Plus2SpatialThetaView cosine;
  teuk::Plus2SpatialThetaView sine;
  Kokkos::View<double*, Kokkos::HostSpace> hc, hs;

  explicit Angles(const std::size_t count)
      : cosine("plus2_value_cos", count),
        sine("plus2_value_sin", count),
        hc(Kokkos::create_mirror_view(cosine)),
        hs(Kokkos::create_mirror_view(sine)) {
    for (std::size_t i = 0; i < count; ++i) {
      hc(i) = -0.52 + 0.71 * static_cast<double>(i) /
                         static_cast<double>(count);
      hs(i) = std::sqrt(1.0 - hc(i) * hc(i));
    }
    Kokkos::deep_copy(cosine, hc);
    Kokkos::deep_copy(sine, hs);
  }
};

teuk::Plus2OrderedPairFieldsT<J> load_fields(
    const teuk::ModeRegistry& registry, const teuk::ModePair& pair,
    const Inputs& in, const std::size_t radial, const std::size_t theta) {
  const std::size_t m1 = registry.index(pair.m1);
  const std::size_t m2 = registry.index(pair.m2);
  const std::size_t sharp = registry.sharp_index(pair.m1);
  const auto x = [&](const std::size_t mode,
                     const teuk::Plus2SpatialPrimitive component) {
    return J{in.hp(mode, p(component), radial, theta),
             in.hpt(mode, p(component), radial, theta)};
  };
  return {x(m1, teuk::Plus2SpatialPrimitive::V),
          x(m1, teuk::Plus2SpatialPrimitive::C),
          teuk::jet_conj(x(sharp, teuk::Plus2SpatialPrimitive::C)),
          x(m1, teuk::Plus2SpatialPrimitive::B),
          teuk::jet_conj(x(sharp, teuk::Plus2SpatialPrimitive::B)),
          x(m1, teuk::Plus2SpatialPrimitive::Sig),
          x(m1, teuk::Plus2SpatialPrimitive::Kap),
          x(m1, teuk::Plus2SpatialPrimitive::Rh),
          teuk::jet_conj(x(sharp, teuk::Plus2SpatialPrimitive::Rh)),
          x(m1, teuk::Plus2SpatialPrimitive::Ep),
          teuk::jet_conj(x(sharp, teuk::Plus2SpatialPrimitive::Ep)),
          x(m2, teuk::Plus2SpatialPrimitive::Z0),
          x(m2, teuk::Plus2SpatialPrimitive::Z1),
          x(m2, teuk::Plus2SpatialPrimitive::H),
          x(m2, teuk::Plus2SpatialPrimitive::Sig),
          x(m2, teuk::Plus2SpatialPrimitive::Kap)};
}

teuk::Plus2OrderedPairJkDerivativesT<J> load_jk(
    const teuk::ModeRegistry& registry, const teuk::ModePair& pair,
    const Inputs& in, const std::size_t radial, const std::size_t theta) {
  const std::size_t m1 = registry.index(pair.m1);
  const std::size_t m2 = registry.index(pair.m2);
  const auto x = [&](const std::size_t mode,
                     const teuk::Plus2ProductionJkDerivative component) {
    return J{in.hjk(mode, jkd(component), radial, theta),
             in.hjkt(mode, jkd(component), radial, theta)};
  };
  return {x(m2, teuk::Plus2ProductionJkDerivative::CapitalDelta4Z1),
          x(m2, teuk::Plus2ProductionJkDerivative::EthPrime4Z1),
          x(m1, teuk::Plus2ProductionJkDerivative::CapitalDelta2CSharp),
          x(m1, teuk::Plus2ProductionJkDerivative::EthPrime1BSharp),
          x(m2, teuk::Plus2ProductionJkDerivative::CapitalDelta5Z0),
          x(m2, teuk::Plus2ProductionJkDerivative::Eth5Z0),
          x(m1, teuk::Plus2ProductionJkDerivative::CapitalDelta2C),
          x(m1, teuk::Plus2ProductionJkDerivative::Eth1B),
          x(m1, teuk::Plus2ProductionJkDerivative::CapitalDelta2V),
          x(m1, teuk::Plus2ProductionJkDerivative::Eth2C),
          x(m1, teuk::Plus2ProductionJkDerivative::EthPrime2CSharp)};
}

teuk::Plus2OrderedPairFieldsT<C> values(
    const teuk::Plus2OrderedPairFieldsT<J>& f) {
  return {f.V1.value,       f.C1.value,       f.Csharp1.value,
          f.B1.value,       f.Bsharp1.value,  f.Sig1.value,
          f.Kap1.value,     f.Rh1.value,      f.Rhsharp1.value,
          f.Ep1.value,      f.Epsharp1.value, f.Z0_2.value,
          f.Z1_2.value,     f.H2.value,       f.Sig2.value,
          f.Kap2.value};
}

TEST_CASE("plus2 production value pairs match host oracle and scale quadratically") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 2}, {0});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.74);
  Angles angles(2);
  Inputs input(registry.size(), grid.size(), 2, "plus2_value_inputs");
  Inputs scaled_input(registry.size(), grid.size(), 2,
                      "plus2_value_scaled_inputs");
  input.fill();
  constexpr double amplitude = -1.9;
  scaled_input.fill(amplitude);
  teuk::Plus2SourceValueSpatialWorkspace workspace(
      registry, grid.size(), 2, "plus2_value_workspace");
  teuk::Plus2SourceValueSpatialWorkspace scaled_workspace(
      registry, grid.size(), 2, "plus2_value_scaled_workspace");
  const teuk::KerrParameters parameters{1.0, 0.88, 1.3};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_plus2_production_ordered_pair_values(
      execution, grid, parameters, angles.cosine, angles.sine,
      input.primitive, input.primitive_tangent, input.jk, input.jk_tangent,
      input.q, workspace);
  teuk::evaluate_plus2_production_ordered_pair_values(
      execution, grid, parameters, angles.cosine, angles.sine,
      scaled_input.primitive, scaled_input.primitive_tangent,
      scaled_input.jk, scaled_input.jk_tangent, scaled_input.q,
      scaled_workspace);
  execution.fence("plus2 production value oracle");
  const auto pair_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.pair_family_value());
  const auto sums = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.summed_value());
  const auto jk_tangents = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.summed_jk_tangent());
  const auto scaled_pairs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, scaled_workspace.pair_family_value());
  const auto scaled_jk_tangents = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, scaled_workspace.summed_jk_tangent());

  const std::size_t radial = 3;
  const std::size_t theta = 1;
  const double radius = grid.coordinate(radial);
  const auto background = teuk::kerr_background_point(
      parameters, radius, angles.hc(theta), angles.hs(theta));
  std::array<J, 2> expected_jk{};
  C expected_q{};
  for (std::size_t pair_index = 0;
       pair_index < registry.ordered_pairs().size(); ++pair_index) {
    const auto& pair = registry.ordered_pairs()[pair_index];
    const auto fields = load_fields(registry, pair, input, radial, theta);
    const auto derivatives = load_jk(registry, pair, input, radial, theta);
    const auto jk = teuk::plus2_compact_ordered_pair_jk_source(
        radius, background, fields, derivatives);
    const std::size_t m2 = registry.index(pair.m2);
    const auto q = teuk::plus2_compact_ordered_pair_q_source(
        radius, background, values(fields),
        teuk::Plus2OrderedPairQDerivativesT<C>{
            derivatives.ethprime1_Bsharp1.value,
            input.hq(m2,
                     qd(teuk::Plus2ProductionQDerivative::CapitalDelta2Sig),
                     radial, theta),
            input.hq(m2,
                     qd(teuk::Plus2ProductionQDerivative::CapitalDelta3Kap),
                     radial, theta),
            input.hq(m2, qd(teuk::Plus2ProductionQDerivative::EthPrime3Kap),
                     radial, theta)});
    const C expected[] = {
        jk.C12.total().value, jk.B12.total().value,
        jk.D12.total().value, q.Er12.total(), q.Et12.total(),
        jk.J12.total().value, jk.K12.total().value, q.Q12.total()};
    for (std::size_t family = 0; family < 8; ++family) {
      CHECK_COMPLEX_NEAR(pair_values(pair_index, family, radial, theta),
                         expected[family], 5.0e-12);
      CHECK_COMPLEX_NEAR(
          scaled_pairs(pair_index, family, radial, theta),
          amplitude * amplitude * expected[family], 2.0e-11);
    }
    expected_jk[0] += jk.J12.total();
    expected_jk[1] += jk.K12.total();
    expected_q += q.Q12.total();
  }
  const std::size_t target = registry.index(0);
  CHECK_COMPLEX_NEAR(sums(target, agg(teuk::Plus2SpatialAggregate::J), radial,
                          theta),
                     expected_jk[0].value, 6.0e-12);
  CHECK_COMPLEX_NEAR(sums(target, agg(teuk::Plus2SpatialAggregate::K), radial,
                          theta),
                     expected_jk[1].value, 6.0e-12);
  CHECK_COMPLEX_NEAR(sums(target, agg(teuk::Plus2SpatialAggregate::Q), radial,
                          theta),
                     expected_q, 6.0e-12);
  CHECK_COMPLEX_NEAR(
      jk_tangents(target, jkagg(teuk::Plus2ProductionJkAggregate::J), radial,
                  theta),
      expected_jk[0].dt, 7.0e-12);
  CHECK_COMPLEX_NEAR(
      jk_tangents(target, jkagg(teuk::Plus2ProductionJkAggregate::K), radial,
                  theta),
      expected_jk[1].dt, 7.0e-12);
  CHECK_COMPLEX_NEAR(
      scaled_jk_tangents(
          target, jkagg(teuk::Plus2ProductionJkAggregate::J), radial, theta),
      amplitude * amplitude * expected_jk[0].dt, 3.0e-11);

  // Non-target parent slots remain exactly zero.
  CHECK_COMPLEX_NEAR(sums(registry.index(-2), 0, radial, theta), C{}, 0.0);
  CHECK(registry.modes()[registry.sharp_index(2)] == -2);
}

TEST_CASE("plus2 production J K tangents match central differences") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 2}, {0});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.74);
  Angles angles(2);
  Inputs base(registry.size(), grid.size(), 2, "plus2_value_fd_base");
  Inputs plus(registry.size(), grid.size(), 2, "plus2_value_fd_plus");
  Inputs minus(registry.size(), grid.size(), 2, "plus2_value_fd_minus");
  base.fill();
  plus.fill();
  minus.fill();
  constexpr double epsilon = 2.0e-6;
  for (std::size_t mode = 0; mode < base.primitive.extent(0); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 2; ++theta) {
        for (std::size_t component = 0; component < base.primitive.extent(1);
             ++component) {
          plus.hp(mode, component, radial, theta) =
              base.hp(mode, component, radial, theta) +
              epsilon * base.hpt(mode, component, radial, theta);
          minus.hp(mode, component, radial, theta) =
              base.hp(mode, component, radial, theta) -
              epsilon * base.hpt(mode, component, radial, theta);
        }
        for (std::size_t component = 0; component < base.jk.extent(1);
             ++component) {
          plus.hjk(mode, component, radial, theta) =
              base.hjk(mode, component, radial, theta) +
              epsilon * base.hjkt(mode, component, radial, theta);
          minus.hjk(mode, component, radial, theta) =
              base.hjk(mode, component, radial, theta) -
              epsilon * base.hjkt(mode, component, radial, theta);
        }
      }
    }
  }
  Kokkos::deep_copy(plus.primitive, plus.hp);
  Kokkos::deep_copy(minus.primitive, minus.hp);
  Kokkos::deep_copy(plus.jk, plus.hjk);
  Kokkos::deep_copy(minus.jk, minus.hjk);
  teuk::Plus2SourceValueSpatialWorkspace wb(
      registry, grid.size(), 2, "plus2_value_fd_wb");
  teuk::Plus2SourceValueSpatialWorkspace wp(
      registry, grid.size(), 2, "plus2_value_fd_wp");
  teuk::Plus2SourceValueSpatialWorkspace wm(
      registry, grid.size(), 2, "plus2_value_fd_wm");
  const teuk::KerrParameters parameters{1.0, -0.79, 1.25};
  const teuk::ExecutionSpace execution;
  const auto evaluate = [&](const Inputs& in,
                            teuk::Plus2SourceValueSpatialWorkspace& w) {
    teuk::evaluate_plus2_production_ordered_pair_values(
        execution, grid, parameters, angles.cosine, angles.sine, in.primitive,
        in.primitive_tangent, in.jk, in.jk_tangent, in.q, w);
  };
  evaluate(base, wb);
  evaluate(plus, wp);
  evaluate(minus, wm);
  execution.fence("plus2 production J K central difference");
  const auto tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wb.summed_jk_tangent());
  const auto plus_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wp.summed_value());
  const auto minus_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wm.summed_value());
  const std::size_t target = registry.index(0);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    for (std::size_t theta = 0; theta < 2; ++theta) {
      for (std::size_t component = 0; component < 2; ++component) {
        CHECK_COMPLEX_NEAR(
            tangent(target, component, radial, theta),
            (plus_value(target, component, radial, theta) -
             minus_value(target, component, radial, theta)) /
                (2.0 * epsilon),
            3.0e-9);
      }
    }
  }
}

TEST_CASE("plus2 production outer source is value-only and applies activation once") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 0, 2}, {0});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.71);
  Angles angles(2);
  teuk::Plus2SourceValueSpatialWorkspace workspace(
      registry, grid.size(), 2, "plus2_value_outer");
  teuk::Plus2SpatialAggregateView sums("plus2_value_outer_sums",
                                        registry.size(), 3, grid.size(), 2);
  teuk::Plus2SpatialOuterDerivativeView derivatives(
      "plus2_value_outer_derivatives", registry.size(), 2, grid.size(), 2);
  Host4 hs(Kokkos::create_mirror_view(sums));
  Host4 hd(Kokkos::create_mirror_view(derivatives));
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 2; ++theta) {
        for (std::size_t component = 0; component < 3; ++component) {
          const double tag = 0.2 + 0.1 * mode + 0.03 * radial -
                             0.02 * theta + 0.04 * component;
          hs(mode, component, radial, theta) = C(tag, -0.2 * tag);
        }
        for (std::size_t component = 0; component < 2; ++component) {
          const double tag = -0.1 + 0.07 * mode - 0.02 * radial +
                             0.03 * theta + 0.05 * component;
          hd(mode, component, radial, theta) = C(tag, 0.3 * tag);
        }
      }
    }
  }
  Kokkos::deep_copy(sums, hs);
  Kokkos::deep_copy(derivatives, hd);
  const teuk::KerrParameters parameters{1.0, -0.94, 1.2};
  constexpr double activation = 0.43;
  const teuk::ExecutionSpace execution;
  teuk::evaluate_plus2_production_outer_source_value(
      execution, grid, parameters, angles.cosine, angles.sine, sums,
      derivatives, activation, workspace);
  execution.fence("plus2 production outer value oracle");
  const auto source = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.source_value());
  const auto forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.forcing_value());
  const std::size_t mode = registry.index(0), radial = 4, theta = 1;
  const double radius = grid.coordinate(radial);
  const auto background = teuk::kerr_background_point(
      parameters, radius, angles.hc(theta), angles.hs(theta));
  const C expected_source =
      activation *
      teuk::plus2_compact_outer_source_over_r6(
          radius, background,
          hs(mode, agg(teuk::Plus2SpatialAggregate::J), radial, theta),
          hs(mode, agg(teuk::Plus2SpatialAggregate::K), radial, theta),
          hs(mode, agg(teuk::Plus2SpatialAggregate::Q), radial, theta),
          teuk::Plus2OuterDerivativesT<C>{
              hd(mode, outer(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                 radial, theta),
              hd(mode, outer(teuk::Plus2SpatialOuterDerivative::Eth6K),
                 radial, theta)})
          .total();
  CHECK_COMPLEX_NEAR(source(mode, radial, theta), expected_source, 3.0e-12);
  CHECK_COMPLEX_NEAR(
      forcing(mode, radial, theta),
      teuk::plus2_coordinate_forcing_from_source_over_r6(
          radius, angles.hc(theta), parameters.spin,
          parameters.compactification_length, expected_source),
      4.0e-12);
}

int allocations = 0;
int deep_copies = 0;
int fences = 0;
void count_allocation(Kokkos::Tools::SpaceHandle, const char*, const void*,
                      std::uint64_t) {
  ++allocations;
}
void count_deep_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                     Kokkos::Tools::SpaceHandle, const char*, const void*,
                     std::uint64_t) {
  ++deep_copies;
}
void count_fence(const char*, std::uint32_t, std::uint64_t*) { ++fences; }

TEST_CASE("plus2 production value source validates shape alias and allocates nothing") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 0, 2}, {0});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.7);
  Angles angles(2);
  Inputs input(registry.size(), grid.size(), 2, "plus2_value_validation");
  input.fill();
  teuk::Plus2SourceValueSpatialWorkspace workspace(
      registry, grid.size(), 2, "plus2_value_validation_workspace");
  const teuk::KerrParameters parameters{1.0, 0.6, 1.2};
  const teuk::ExecutionSpace execution;
  bool alias_rejected = false;
  try {
    teuk::evaluate_plus2_production_ordered_pair_values(
        execution, grid, parameters, angles.cosine, angles.sine,
        input.primitive, input.primitive, input.jk, input.jk_tangent, input.q,
        workspace);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected);
  bool shape_rejected = false;
  try {
    teuk::Plus2ProductionQDerivativeView wrong(
        "plus2_value_wrong_q", registry.size(), 4, grid.size(), 2);
    teuk::evaluate_plus2_production_ordered_pair_values(
        execution, grid, parameters, angles.cosine, angles.sine,
        input.primitive, input.primitive_tangent, input.jk, input.jk_tangent,
        wrong, workspace);
  } catch (const std::invalid_argument&) {
    shape_rejected = true;
  }
  CHECK(shape_rejected);

  // Warm up the named kernel before observing evaluation callbacks.
  teuk::evaluate_plus2_production_ordered_pair_values(
      execution, grid, parameters, angles.cosine, angles.sine,
      input.primitive, input.primitive_tangent, input.jk, input.jk_tangent,
      input.q, workspace);
  execution.fence("plus2 production value warmup");
  teuk::Plus2SpatialOuterDerivativeView outer_derivatives(
      "plus2_value_validation_outer", registry.size(), 2, grid.size(), 2);
  Kokkos::deep_copy(outer_derivatives, C(0.1, -0.2));
  allocations = 0;
  deep_copies = 0;
  fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(count_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(count_deep_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_fence);
  teuk::evaluate_plus2_production_ordered_pair_values(
      execution, grid, parameters, angles.cosine, angles.sine,
      input.primitive, input.primitive_tangent, input.jk, input.jk_tangent,
      input.q, workspace);
  teuk::evaluate_plus2_production_outer_source_value(
      execution, grid, parameters, angles.cosine, angles.sine,
      workspace.summed_value(), outer_derivatives, 1.0, workspace);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  execution.fence("plus2 production value no allocation");
  CHECK(allocations == 0);
  CHECK(deep_copies == 0);
  CHECK(fences == 0);
}

}  // namespace
