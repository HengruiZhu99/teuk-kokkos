#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// Deliberately compose all three public gates in this order.
#include "teuk/plus2_linear_psi0.hpp"
#include "teuk/plus2_source.hpp"
#include "teuk/plus2_source_primitives.hpp"
#include "teuk/plus2_source_spatial.hpp"

namespace {

using C = teuk::Complex;
using J = teuk::Jet1<C>;
using Host4 =
    Kokkos::View<C****, Kokkos::LayoutRight, Kokkos::HostSpace>;

int spatial_allocations = 0;
int spatial_deep_copies = 0;
int spatial_fences = 0;

void count_spatial_allocation(Kokkos::Tools::SpaceHandle, const char*,
                              const void*, std::uint64_t) {
  ++spatial_allocations;
}

void count_spatial_deep_copy(Kokkos::Tools::SpaceHandle, const char*,
                             const void*, Kokkos::Tools::SpaceHandle,
                             const char*, const void*, std::uint64_t) {
  ++spatial_deep_copies;
}

void count_spatial_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++spatial_fences;
}

constexpr std::size_t p(const teuk::Plus2SpatialPrimitive component) {
  return static_cast<std::size_t>(component);
}

constexpr std::size_t d(
    const teuk::Plus2SpatialPairDerivative component) {
  return static_cast<std::size_t>(component);
}

constexpr std::size_t f(const teuk::Plus2SpatialPairFamily family) {
  return static_cast<std::size_t>(family);
}

constexpr std::size_t a(const teuk::Plus2SpatialAggregate aggregate) {
  return static_cast<std::size_t>(aggregate);
}

constexpr std::size_t od(
    const teuk::Plus2SpatialOuterDerivative derivative) {
  return static_cast<std::size_t>(derivative);
}

struct SpatialInputs {
  teuk::Plus2SpatialPrimitiveView primitive_value;
  teuk::Plus2SpatialPrimitiveView primitive_tangent;
  teuk::Plus2SpatialPairDerivativeView derivative_value;
  teuk::Plus2SpatialPairDerivativeView derivative_tangent;
  Host4 host_primitive_value;
  Host4 host_primitive_tangent;
  Host4 host_derivative_value;
  Host4 host_derivative_tangent;

  SpatialInputs(const std::size_t modes, const std::size_t radial,
                const std::size_t theta, const std::string& label)
      : primitive_value(
            label + "_primitive_value", modes,
            static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count),
            radial, theta),
        primitive_tangent(
            label + "_primitive_tangent", modes,
            static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count),
            radial, theta),
        derivative_value(
            label + "_derivative_value", modes,
            static_cast<std::size_t>(
                teuk::Plus2SpatialPairDerivative::Count),
            radial, theta),
        derivative_tangent(
            label + "_derivative_tangent", modes,
            static_cast<std::size_t>(
                teuk::Plus2SpatialPairDerivative::Count),
            radial, theta),
        host_primitive_value(
            label + "_host_primitive_value", modes,
            static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count),
            radial, theta),
        host_primitive_tangent(
            label + "_host_primitive_tangent", modes,
            static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count),
            radial, theta),
        host_derivative_value(
            label + "_host_derivative_value", modes,
            static_cast<std::size_t>(
                teuk::Plus2SpatialPairDerivative::Count),
            radial, theta),
        host_derivative_tangent(
            label + "_host_derivative_tangent", modes,
            static_cast<std::size_t>(
                teuk::Plus2SpatialPairDerivative::Count),
            radial, theta) {}

  void fill(const double scale = 1.0) {
    for (std::size_t mode = 0; mode < primitive_value.extent(0); ++mode) {
      for (std::size_t component = 0;
           component < primitive_value.extent(1); ++component) {
        for (std::size_t radial = 0; radial < primitive_value.extent(2);
             ++radial) {
          for (std::size_t theta = 0; theta < primitive_value.extent(3);
               ++theta) {
            const double tag = 0.31 + 0.19 * mode + 0.023 * component +
                               0.011 * radial - 0.007 * theta;
            host_primitive_value(mode, component, radial, theta) =
                scale * C(tag, -0.47 * tag + 0.03 * mode);
            host_primitive_tangent(mode, component, radial, theta) =
                scale * C(-0.13 * tag + 0.02 * theta,
                          0.21 * tag - 0.01 * component);
          }
        }
      }
      for (std::size_t component = 0;
           component < derivative_value.extent(1); ++component) {
        for (std::size_t radial = 0; radial < derivative_value.extent(2);
             ++radial) {
          for (std::size_t theta = 0; theta < derivative_value.extent(3);
               ++theta) {
            const double tag = -0.28 + 0.17 * mode - 0.019 * component +
                               0.013 * radial + 0.005 * theta;
            host_derivative_value(mode, component, radial, theta) =
                scale * C(tag, 0.32 * tag - 0.02 * mode);
            host_derivative_tangent(mode, component, radial, theta) =
                scale * C(0.16 * tag + 0.01 * component,
                          -0.24 * tag + 0.03 * theta);
          }
        }
      }
    }
    sync();
  }

  void sync() const {
    Kokkos::deep_copy(primitive_value, host_primitive_value);
    Kokkos::deep_copy(primitive_tangent, host_primitive_tangent);
    Kokkos::deep_copy(derivative_value, host_derivative_value);
    Kokkos::deep_copy(derivative_tangent, host_derivative_tangent);
  }
};

struct ThetaInputs {
  teuk::Plus2SpatialThetaView cosine;
  teuk::Plus2SpatialThetaView sine;
  Kokkos::View<double*, Kokkos::HostSpace> host_cosine;
  Kokkos::View<double*, Kokkos::HostSpace> host_sine;

  explicit ThetaInputs(const std::size_t count)
      : cosine("plus2_spatial_cosine", count),
        sine("plus2_spatial_sine", count),
        host_cosine("plus2_spatial_host_cosine", count),
        host_sine("plus2_spatial_host_sine", count) {
    for (std::size_t theta = 0; theta < count; ++theta) {
      host_cosine(theta) = -0.71 + 1.42 * static_cast<double>(theta + 1) /
                                      static_cast<double>(count + 1);
      host_sine(theta) =
          std::sqrt(1.0 - host_cosine(theta) * host_cosine(theta));
    }
    Kokkos::deep_copy(cosine, host_cosine);
    Kokkos::deep_copy(sine, host_sine);
  }
};

J host_primitive(const SpatialInputs& input, const std::size_t mode,
                 const teuk::Plus2SpatialPrimitive component,
                 const std::size_t radial, const std::size_t theta) {
  return {input.host_primitive_value(mode, p(component), radial, theta),
          input.host_primitive_tangent(mode, p(component), radial, theta)};
}

J host_derivative(const SpatialInputs& input, const std::size_t mode,
                  const teuk::Plus2SpatialPairDerivative component,
                  const std::size_t radial, const std::size_t theta) {
  return {input.host_derivative_value(mode, d(component), radial, theta),
          input.host_derivative_tangent(mode, d(component), radial, theta)};
}

teuk::Plus2OrderedPairSourceT<J> host_pair_oracle(
    const teuk::ModeRegistry& registry, const teuk::ModePair& pair,
    const SpatialInputs& input, const double radius,
    const teuk::KerrBackgroundPoint& background, const std::size_t radial,
    const std::size_t theta, const bool erroneous_same_mode_sharp = false) {
  const auto lookup = teuk::make_plus2_pair_lookup(registry, pair);
  const std::size_t m1 = lookup.index1;
  const std::size_t m2 = lookup.index2;
  const std::size_t sharp1 =
      erroneous_same_mode_sharp ? lookup.index1 : lookup.sharp1;
  const teuk::Plus2OrderedPairFieldsT<J> fields{
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::V, radial, theta),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::C, radial, theta),
      teuk::jet_conj(host_primitive(
          input, sharp1, teuk::Plus2SpatialPrimitive::C, radial, theta)),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::B, radial, theta),
      teuk::jet_conj(host_primitive(
          input, sharp1, teuk::Plus2SpatialPrimitive::B, radial, theta)),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::Sig, radial,
                     theta),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::Kap, radial,
                     theta),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::Rh, radial,
                     theta),
      teuk::jet_conj(host_primitive(
          input, sharp1, teuk::Plus2SpatialPrimitive::Rh, radial, theta)),
      host_primitive(input, m1, teuk::Plus2SpatialPrimitive::Ep, radial,
                     theta),
      teuk::jet_conj(host_primitive(
          input, sharp1, teuk::Plus2SpatialPrimitive::Ep, radial, theta)),
      host_primitive(input, m2, teuk::Plus2SpatialPrimitive::Z0, radial,
                     theta),
      host_primitive(input, m2, teuk::Plus2SpatialPrimitive::Z1, radial,
                     theta),
      host_primitive(input, m2, teuk::Plus2SpatialPrimitive::H, radial, theta),
      host_primitive(input, m2, teuk::Plus2SpatialPrimitive::Sig, radial,
                     theta),
      host_primitive(input, m2, teuk::Plus2SpatialPrimitive::Kap, radial,
                     theta)};
  const teuk::Plus2OrderedPairDerivativesT<J> derivatives{
      host_derivative(input, m2,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta4Z1,
                      radial, theta),
      host_derivative(input, m2, teuk::Plus2SpatialPairDerivative::EthPrime4Z1,
                      radial, theta),
      host_derivative(
          input, m1,
          teuk::Plus2SpatialPairDerivative::CapitalDelta2CSharp, radial,
          theta),
      host_derivative(input, m1,
                      teuk::Plus2SpatialPairDerivative::EthPrime1BSharp,
                      radial, theta),
      host_derivative(input, m2,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta5Z0,
                      radial, theta),
      host_derivative(input, m2, teuk::Plus2SpatialPairDerivative::Eth5Z0,
                      radial, theta),
      host_derivative(input, m1,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta2C,
                      radial, theta),
      host_derivative(input, m1, teuk::Plus2SpatialPairDerivative::Eth1B,
                      radial, theta),
      host_derivative(input, m1,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta2V,
                      radial, theta),
      host_derivative(input, m1, teuk::Plus2SpatialPairDerivative::Eth2C,
                      radial, theta),
      host_derivative(
          input, m1,
          teuk::Plus2SpatialPairDerivative::EthPrime2CSharp, radial, theta),
      host_derivative(input, m2,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta2Sig,
                      radial, theta),
      host_derivative(input, m2,
                      teuk::Plus2SpatialPairDerivative::CapitalDelta3Kap,
                      radial, theta),
      host_derivative(input, m2,
                      teuk::Plus2SpatialPairDerivative::EthPrime3Kap, radial,
                      theta)};
  return teuk::plus2_compact_ordered_pair_source(radius, background, fields,
                                                   derivatives);
}

std::array<J, 8> family_totals(
    const teuk::Plus2OrderedPairSourceT<J>& source) {
  return {source.C12.total(), source.B12.total(), source.D12.total(),
          source.Er12.total(), source.Et12.total(), source.J12.total(),
          source.K12.total(), source.Q12.total()};
}

TEST_CASE("plus2 spatial pair kernel closes every family against host oracle") {
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-2, 0, 2},
                                    {-4, 0, 4});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.83);
  ThetaInputs angles(3);
  SpatialInputs input(registry.size(), grid.size(), angles.cosine.extent(0),
                      "plus2_spatial_oracle");
  input.fill();
  teuk::Plus2SourceSpatialWorkspace workspace(
      registry, grid.size(), angles.cosine.extent(0), "plus2_spatial_oracle");
  const teuk::KerrParameters parameters{1.0, 0.79, 1.3};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_plus2_spatial_ordered_pairs(
      execution, grid, parameters, angles.cosine, angles.sine,
      input.primitive_value, input.primitive_tangent, input.derivative_value,
      input.derivative_tangent, workspace);
  execution.fence("plus2 spatial pair host oracle");

  const auto pair_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.pair_family_value());
  const auto pair_tangents = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.pair_family_tangent());
  const auto sums = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.summed_value());
  const auto sum_tangents = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.summed_tangent());
  const auto stored_pairs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.pair_lookup());
  const auto stored_targets = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.target_indices());
  const auto stored_offsets = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.pair_offsets());
  CHECK(stored_pairs.extent(0) == registry.ordered_pairs().size());
  for (std::size_t pair = 0; pair < registry.ordered_pairs().size(); ++pair) {
    const auto expected = teuk::make_plus2_pair_lookup(
        registry, registry.ordered_pairs()[pair]);
    CHECK(stored_pairs(pair).m1 == expected.m1);
    CHECK(stored_pairs(pair).m2 == expected.m2);
    CHECK(stored_pairs(pair).target == expected.target);
    CHECK(stored_pairs(pair).sharp1 == expected.sharp1);
    CHECK(stored_pairs(pair).sharp2 == expected.sharp2);
  }
  for (std::size_t target = 0; target < registry.targets().size(); ++target) {
    const auto [begin, end] = registry.pair_range(registry.targets()[target]);
    CHECK(stored_targets(target) == registry.index(registry.targets()[target]));
    CHECK(stored_offsets(target) == begin);
    CHECK(stored_offsets(target + 1) == end);
  }
  for (const int target_mode : registry.targets()) {
    const auto [begin, end] = registry.pair_range(target_mode);
    const std::size_t target = registry.index(target_mode);
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < angles.cosine.extent(0); ++theta) {
        std::array<J, 3> expected_sums{};
        const double radius = grid.coordinate(radial);
        const auto background = teuk::kerr_background_point(
            parameters, radius, angles.host_cosine(theta),
            angles.host_sine(theta));
        for (std::size_t pair = begin; pair < end; ++pair) {
          const auto expected = family_totals(host_pair_oracle(
              registry, registry.ordered_pairs()[pair], input, radius,
              background, radial, theta));
          for (std::size_t family = 0; family < expected.size(); ++family) {
            CHECK_COMPLEX_NEAR(pair_values(pair, family, radial, theta),
                               expected[family].value, 3.0e-12);
            CHECK_COMPLEX_NEAR(pair_tangents(pair, family, radial, theta),
                               expected[family].dt, 4.0e-12);
          }
          expected_sums[a(teuk::Plus2SpatialAggregate::J)] +=
              expected[f(teuk::Plus2SpatialPairFamily::J)];
          expected_sums[a(teuk::Plus2SpatialAggregate::K)] +=
              expected[f(teuk::Plus2SpatialPairFamily::K)];
          expected_sums[a(teuk::Plus2SpatialAggregate::Q)] +=
              expected[f(teuk::Plus2SpatialPairFamily::Q)];
        }
        for (std::size_t aggregate = 0; aggregate < expected_sums.size();
             ++aggregate) {
          CHECK_COMPLEX_NEAR(sums(target, aggregate, radial, theta),
                             expected_sums[aggregate].value, 5.0e-12);
          CHECK_COMPLEX_NEAR(sum_tangents(target, aggregate, radial, theta),
                             expected_sums[aggregate].dt, 6.0e-12);
        }
      }
    }
  }

  // Stored but unconfigured target modes remain exactly zero.
  for (const int mode : {-2, 2}) {
    const std::size_t index = registry.index(mode);
    CHECK_COMPLEX_NEAR(sums(index, 0, 2, 1), C{}, 0.0);
    CHECK_COMPLEX_NEAR(sum_tangents(index, 2, 2, 1), C{}, 0.0);
  }
  const auto lookup = teuk::make_plus2_pair_lookup(
      registry, registry.ordered_pairs().front());
  CHECK(registry.modes()[lookup.sharp1] == -lookup.m1);
  CHECK(Kokkos::abs(input.host_primitive_value(
                        lookup.sharp1, p(teuk::Plus2SpatialPrimitive::C), 2,
                        1) -
                    input.host_primitive_value(
                        lookup.index1, p(teuk::Plus2SpatialPrimitive::C), 2,
                        1)) > 0.1);
  const double sharp_radius = grid.coordinate(2);
  const auto sharp_background = teuk::kerr_background_point(
      parameters, sharp_radius, angles.host_cosine(1), angles.host_sine(1));
  const auto correct_sharp = family_totals(host_pair_oracle(
      registry, registry.ordered_pairs().front(), input, sharp_radius,
      sharp_background, 2, 1));
  const auto erroneous_same_mode = family_totals(host_pair_oracle(
      registry, registry.ordered_pairs().front(), input, sharp_radius,
      sharp_background, 2, 1, true));
  CHECK(Kokkos::abs(correct_sharp[f(teuk::Plus2SpatialPairFamily::C)].value -
                    erroneous_same_mode[
                        f(teuk::Plus2SpatialPairFamily::C)].value) >
        1.0e-4);
}

void shift_inputs(const SpatialInputs& base, SpatialInputs& shifted,
                  const double epsilon) {
  for (std::size_t mode = 0; mode < base.primitive_value.extent(0); ++mode) {
    for (std::size_t component = 0;
         component < base.primitive_value.extent(1); ++component) {
      for (std::size_t radial = 0; radial < base.primitive_value.extent(2);
           ++radial) {
        for (std::size_t theta = 0; theta < base.primitive_value.extent(3);
             ++theta) {
          shifted.host_primitive_value(mode, component, radial, theta) =
              base.host_primitive_value(mode, component, radial, theta) +
              epsilon *
                  base.host_primitive_tangent(mode, component, radial, theta);
          shifted.host_primitive_tangent(mode, component, radial, theta) =
              base.host_primitive_tangent(mode, component, radial, theta);
        }
      }
    }
    for (std::size_t component = 0;
         component < base.derivative_value.extent(1); ++component) {
      for (std::size_t radial = 0; radial < base.derivative_value.extent(2);
           ++radial) {
        for (std::size_t theta = 0; theta < base.derivative_value.extent(3);
             ++theta) {
          shifted.host_derivative_value(mode, component, radial, theta) =
              base.host_derivative_value(mode, component, radial, theta) +
              epsilon * base.host_derivative_tangent(mode, component, radial,
                                                      theta);
          shifted.host_derivative_tangent(mode, component, radial, theta) =
              base.host_derivative_tangent(mode, component, radial, theta);
        }
      }
    }
  }
  shifted.sync();
}

TEST_CASE("plus2 spatial pair tangents scale quadratically and match FD") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 2}, {0});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.72);
  ThetaInputs angles(2);
  SpatialInputs base(registry.size(), grid.size(), 2, "plus2_pair_base");
  SpatialInputs plus(registry.size(), grid.size(), 2, "plus2_pair_plus");
  SpatialInputs minus(registry.size(), grid.size(), 2, "plus2_pair_minus");
  SpatialInputs scaled(registry.size(), grid.size(), 2, "plus2_pair_scaled");
  base.fill();
  constexpr double epsilon = 2.0e-6;
  shift_inputs(base, plus, epsilon);
  shift_inputs(base, minus, -epsilon);
  constexpr double amplitude = -2.4;
  scaled.fill(amplitude);
  teuk::Plus2SourceSpatialWorkspace wbase(registry, grid.size(), 2,
                                           "plus2_pair_wbase");
  teuk::Plus2SourceSpatialWorkspace wplus(registry, grid.size(), 2,
                                           "plus2_pair_wplus");
  teuk::Plus2SourceSpatialWorkspace wminus(registry, grid.size(), 2,
                                            "plus2_pair_wminus");
  teuk::Plus2SourceSpatialWorkspace wscaled(registry, grid.size(), 2,
                                             "plus2_pair_wscaled");
  const teuk::KerrParameters parameters{1.0, -0.63, 1.4};
  const teuk::ExecutionSpace execution;
  const auto evaluate = [&](const SpatialInputs& input,
                            teuk::Plus2SourceSpatialWorkspace& workspace) {
    teuk::evaluate_plus2_spatial_ordered_pairs(
        execution, grid, parameters, angles.cosine, angles.sine,
        input.primitive_value, input.primitive_tangent,
        input.derivative_value, input.derivative_tangent, workspace);
  };
  evaluate(base, wbase);
  evaluate(plus, wplus);
  evaluate(minus, wminus);
  evaluate(scaled, wscaled);
  execution.fence("plus2 pair tangent comparisons");
  const auto value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wbase.pair_family_value());
  const auto tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wbase.pair_family_tangent());
  const auto plus_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wplus.pair_family_value());
  const auto minus_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wminus.pair_family_value());
  const auto scaled_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wscaled.pair_family_value());
  const auto scaled_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wscaled.pair_family_tangent());
  const auto sum_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wbase.summed_value());
  const auto sum_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wbase.summed_tangent());
  const auto scaled_sum_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wscaled.summed_value());
  const auto scaled_sum_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wscaled.summed_tangent());
  for (std::size_t pair = 0; pair < wbase.pair_count(); ++pair) {
    for (std::size_t family = 0;
         family <
         static_cast<std::size_t>(teuk::Plus2SpatialPairFamily::Count);
         ++family) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < 2; ++theta) {
          const C finite_difference =
              (plus_value(pair, family, radial, theta) -
               minus_value(pair, family, radial, theta)) /
              (2.0 * epsilon);
          CHECK_COMPLEX_NEAR(tangent(pair, family, radial, theta),
                             finite_difference, 3.0e-9);
          CHECK_COMPLEX_NEAR(
              scaled_value(pair, family, radial, theta),
              amplitude * amplitude * value(pair, family, radial, theta),
              8.0e-12);
          CHECK_COMPLEX_NEAR(
              scaled_tangent(pair, family, radial, theta),
              amplitude * amplitude * tangent(pair, family, radial, theta),
              1.0e-11);
        }
      }
    }
  }
  const std::size_t target = registry.index(0);
  for (std::size_t aggregate = 0;
       aggregate <
       static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count);
       ++aggregate) {
    CHECK_COMPLEX_NEAR(scaled_sum_value(target, aggregate, 3, 1),
                       amplitude * amplitude *
                           sum_value(target, aggregate, 3, 1),
                       1.0e-11);
    CHECK_COMPLEX_NEAR(scaled_sum_tangent(target, aggregate, 3, 1),
                       amplitude * amplitude *
                           sum_tangent(target, aggregate, 3, 1),
                       1.0e-11);
  }
}

TEST_CASE("plus2 spatial outer kernel matches Jet oracle and activation") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 0, 2}, {-2, 0, 2});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.77);
  ThetaInputs angles(3);
  teuk::Plus2SourceSpatialWorkspace workspace(registry, grid.size(), 3,
                                               "plus2_outer");
  teuk::Plus2SpatialRank4View sums("plus2_outer_sums", registry.size(), 3,
                                    grid.size(), 3);
  teuk::Plus2SpatialRank4View sum_tangents(
      "plus2_outer_sum_tangents", registry.size(), 3, grid.size(), 3);
  teuk::Plus2SpatialRank4View derivatives(
      "plus2_outer_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 3);
  teuk::Plus2SpatialRank4View derivative_tangents(
      "plus2_outer_derivative_tangents", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 3);
  Host4 hs("plus2_outer_host_sums", registry.size(), 3, grid.size(), 3);
  Host4 hst("plus2_outer_host_sum_tangents", registry.size(), 3, grid.size(),
            3);
  Host4 hd("plus2_outer_host_derivatives", registry.size(),
           static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
           grid.size(), 3);
  Host4 hdt(
      "plus2_outer_host_derivative_tangents", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 3);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 3; ++theta) {
        for (std::size_t component = 0; component < 3; ++component) {
          const double tag = 0.4 + 0.2 * mode + 0.03 * radial -
                             0.02 * theta + 0.05 * component;
          hs(mode, component, radial, theta) = C(tag, -0.3 * tag);
          hst(mode, component, radial, theta) = C(0.17 * tag, 0.11 * tag);
        }
        for (std::size_t component = 0;
             component < static_cast<std::size_t>(
                             teuk::Plus2SpatialOuterDerivative::Count);
             ++component) {
          const double tag = -0.2 + 0.1 * mode + 0.04 * radial +
                             0.01 * theta - 0.03 * component;
          hd(mode, component, radial, theta) = C(tag, 0.23 * tag);
          hdt(mode, component, radial, theta) = C(-0.13 * tag, 0.19 * tag);
        }
      }
    }
  }
  Kokkos::deep_copy(sums, hs);
  Kokkos::deep_copy(sum_tangents, hst);
  Kokkos::deep_copy(derivatives, hd);
  Kokkos::deep_copy(derivative_tangents, hdt);
  constexpr double activation = 0.37;
  const teuk::KerrParameters parameters{1.0, 0.92, 1.25};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_plus2_spatial_outer_source(
      execution, grid, parameters, angles.cosine, angles.sine, sums,
      sum_tangents, derivatives, derivative_tangents, activation, workspace);
  execution.fence("plus2 outer oracle");
  const auto source = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.source_over_r6_value());
  const auto source_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.source_over_r6_tangent());
  const auto source_over_r7 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.source_over_r7_value());
  const auto source_over_r7_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.source_over_r7_tangent());
  const auto forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.forcing_value());
  const auto forcing_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.forcing_tangent());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 3; ++theta) {
        const double radius = grid.coordinate(radial);
        const auto background = teuk::kerr_background_point(
            parameters, radius, angles.host_cosine(theta),
            angles.host_sine(theta));
        const auto raw = teuk::plus2_compact_outer_source_over_r6(
            radius, background,
            J(hs(mode, a(teuk::Plus2SpatialAggregate::J), radial, theta),
              hst(mode, a(teuk::Plus2SpatialAggregate::J), radial, theta)),
            J(hs(mode, a(teuk::Plus2SpatialAggregate::K), radial, theta),
              hst(mode, a(teuk::Plus2SpatialAggregate::K), radial, theta)),
            J(hs(mode, a(teuk::Plus2SpatialAggregate::Q), radial, theta),
              hst(mode, a(teuk::Plus2SpatialAggregate::Q), radial, theta)),
            teuk::Plus2OuterDerivativesT<J>{
                J(hd(mode, od(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                     radial, theta),
                  hdt(mode, od(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                      radial, theta)),
                J(hd(mode, od(teuk::Plus2SpatialOuterDerivative::Eth6K),
                     radial, theta),
                  hdt(mode, od(teuk::Plus2SpatialOuterDerivative::Eth6K),
                      radial, theta))});
        const J expected_source = activation * raw.total();
        const auto regularized = teuk::plus2_compact_outer_source_over_r7(
            radius, background,
            J(hs(mode, a(teuk::Plus2SpatialAggregate::K), radial, theta),
              hst(mode, a(teuk::Plus2SpatialAggregate::K), radial, theta)),
            J(hs(mode, a(teuk::Plus2SpatialAggregate::Q), radial, theta),
              hst(mode, a(teuk::Plus2SpatialAggregate::Q), radial, theta)),
            teuk::Plus2RegularizedOuterDerivativesT<J>{
                J(hd(mode,
                     od(teuk::Plus2SpatialOuterDerivative::
                            RegularizedThorn5JMinusOpticalJOverR),
                     radial, theta),
                  hdt(mode,
                      od(teuk::Plus2SpatialOuterDerivative::
                             RegularizedThorn5JMinusOpticalJOverR),
                      radial, theta)),
                J(hd(mode, od(teuk::Plus2SpatialOuterDerivative::Eth6K),
                     radial, theta),
                  hdt(mode, od(teuk::Plus2SpatialOuterDerivative::Eth6K),
                      radial, theta))});
        const J expected_source_over_r7 = activation * regularized.total();
        const J expected_forcing =
            teuk::plus2_coordinate_forcing_from_source_over_r7(
                radius, angles.host_cosine(theta), parameters.spin,
                parameters.compactification_length,
                expected_source_over_r7);
        CHECK_COMPLEX_NEAR(source(mode, radial, theta),
                           expected_source.value, 3.0e-12);
        CHECK_COMPLEX_NEAR(source_tangent(mode, radial, theta),
                           expected_source.dt, 3.0e-12);
        CHECK_COMPLEX_NEAR(source_over_r7(mode, radial, theta),
                           expected_source_over_r7.value, 3.0e-12);
        CHECK_COMPLEX_NEAR(source_over_r7_tangent(mode, radial, theta),
                           expected_source_over_r7.dt, 3.0e-12);
        CHECK_COMPLEX_NEAR(forcing(mode, radial, theta),
                           expected_forcing.value, 4.0e-12);
        CHECK_COMPLEX_NEAR(forcing_tangent(mode, radial, theta),
                           expected_forcing.dt, 4.0e-12);
      }
    }
  }

  // The outer Jet tangent is checked independently by perturbing every
  // projected aggregate and outer-operator slot in its tangent direction.
  constexpr double epsilon = 2.0e-6;
  teuk::Plus2SpatialRank4View plus_sums(
      "plus2_outer_plus_sums", registry.size(), 3, grid.size(), 3);
  teuk::Plus2SpatialRank4View minus_sums(
      "plus2_outer_minus_sums", registry.size(), 3, grid.size(), 3);
  teuk::Plus2SpatialRank4View plus_derivatives(
      "plus2_outer_plus_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 3);
  teuk::Plus2SpatialRank4View minus_derivatives(
      "plus2_outer_minus_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 3);
  Host4 hsp("plus2_outer_host_plus_sums", registry.size(), 3, grid.size(), 3);
  Host4 hsm("plus2_outer_host_minus_sums", registry.size(), 3, grid.size(),
            3);
  Host4 hdp("plus2_outer_host_plus_derivatives", registry.size(),
            static_cast<std::size_t>(
                teuk::Plus2SpatialOuterDerivative::Count),
            grid.size(), 3);
  Host4 hdm("plus2_outer_host_minus_derivatives", registry.size(),
            static_cast<std::size_t>(
                teuk::Plus2SpatialOuterDerivative::Count),
            grid.size(), 3);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 3; ++theta) {
        for (std::size_t component = 0; component < 3; ++component) {
          hsp(mode, component, radial, theta) =
              hs(mode, component, radial, theta) +
              epsilon * hst(mode, component, radial, theta);
          hsm(mode, component, radial, theta) =
              hs(mode, component, radial, theta) -
              epsilon * hst(mode, component, radial, theta);
        }
        for (std::size_t component = 0;
             component < static_cast<std::size_t>(
                             teuk::Plus2SpatialOuterDerivative::Count);
             ++component) {
          hdp(mode, component, radial, theta) =
              hd(mode, component, radial, theta) +
              epsilon * hdt(mode, component, radial, theta);
          hdm(mode, component, radial, theta) =
              hd(mode, component, radial, theta) -
              epsilon * hdt(mode, component, radial, theta);
        }
      }
    }
  }
  Kokkos::deep_copy(plus_sums, hsp);
  Kokkos::deep_copy(minus_sums, hsm);
  Kokkos::deep_copy(plus_derivatives, hdp);
  Kokkos::deep_copy(minus_derivatives, hdm);
  teuk::Plus2SourceSpatialWorkspace plus_workspace(
      registry, grid.size(), 3, "plus2_outer_plus");
  teuk::Plus2SourceSpatialWorkspace minus_workspace(
      registry, grid.size(), 3, "plus2_outer_minus");
  teuk::evaluate_plus2_spatial_outer_source(
      execution, grid, parameters, angles.cosine, angles.sine, plus_sums,
      sum_tangents, plus_derivatives, derivative_tangents, activation,
      plus_workspace);
  teuk::evaluate_plus2_spatial_outer_source(
      execution, grid, parameters, angles.cosine, angles.sine, minus_sums,
      sum_tangents, minus_derivatives, derivative_tangents, activation,
      minus_workspace);
  execution.fence("plus2 outer central difference");
  const auto plus_source = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, plus_workspace.source_over_r6_value());
  const auto minus_source = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, minus_workspace.source_over_r6_value());
  const auto plus_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, plus_workspace.forcing_value());
  const auto minus_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, minus_workspace.forcing_value());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 3; ++theta) {
        CHECK_COMPLEX_NEAR(
            source_tangent(mode, radial, theta),
            (plus_source(mode, radial, theta) -
             minus_source(mode, radial, theta)) /
                (2.0 * epsilon),
            2.0e-9);
        CHECK_COMPLEX_NEAR(
            forcing_tangent(mode, radial, theta),
            (plus_forcing(mode, radial, theta) -
             minus_forcing(mode, radial, theta)) /
                (2.0 * epsilon),
            2.0e-9);
      }
    }
  }

  teuk::evaluate_plus2_spatial_outer_source(
      execution, grid, parameters, angles.cosine, angles.sine, sums,
      sum_tangents, derivatives, derivative_tangents, 0.0, workspace);
  execution.fence("plus2 outer disabled activation");
  const auto disabled = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.forcing_value());
  CHECK_COMPLEX_NEAR(disabled(1, 3, 2), C{}, 0.0);
}

TEST_CASE("plus2 spatial evaluators validate shape and allocate nothing") {
  const teuk::ModeRegistry registry({-2, 0, 2});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.7);
  ThetaInputs angles(2);
  SpatialInputs input(registry.size(), grid.size(), 2, "plus2_noalloc");
  input.fill();
  teuk::Plus2SourceSpatialWorkspace workspace(registry, grid.size(), 2,
                                               "plus2_noalloc");
  const teuk::KerrParameters parameters{1.0, 0.55, 1.2};
  const teuk::ExecutionSpace execution;
  bool shape_rejected = false;
  try {
    teuk::Plus2SpatialRank4View wrong("plus2_wrong_shape", registry.size(), 13,
                                      grid.size(), 2);
    teuk::evaluate_plus2_spatial_ordered_pairs(
        execution, grid, parameters, angles.cosine, angles.sine, wrong,
        input.primitive_tangent, input.derivative_value,
        input.derivative_tangent, workspace);
  } catch (const std::invalid_argument&) {
    shape_rejected = true;
  }
  CHECK(shape_rejected);
  bool sharp_rejected = false;
  try {
    const teuk::ModeRegistry incomplete({0, 2});
    const teuk::Plus2SourceSpatialWorkspace rejected(incomplete, 5, 2);
    static_cast<void>(rejected);
  } catch (const std::invalid_argument&) {
    sharp_rejected = true;
  }
  CHECK(sharp_rejected);
  bool activation_rejected = false;
  try {
    teuk::Plus2SpatialRank4View outer_derivatives(
        "plus2_noalloc_outer_derivatives", registry.size(),
        static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
        grid.size(), 2);
    teuk::evaluate_plus2_spatial_outer_source(
        execution, grid, parameters, angles.cosine, angles.sine,
        workspace.summed_value(), workspace.summed_tangent(),
        outer_derivatives, outer_derivatives,
        std::numeric_limits<double>::quiet_NaN(), workspace);
  } catch (const std::invalid_argument&) {
    activation_rejected = true;
  }
  CHECK(activation_rejected);
  bool outer_shape_rejected = false;
  try {
    teuk::Plus2SpatialRank4View wrong_sums(
        "plus2_wrong_outer_shape", registry.size(), 2, grid.size(), 2);
    teuk::Plus2SpatialRank4View outer_derivatives(
        "plus2_wrong_outer_derivatives", registry.size(), 2, grid.size(), 2);
    teuk::evaluate_plus2_spatial_outer_source(
        execution, grid, parameters, angles.cosine, angles.sine, wrong_sums,
        wrong_sums, outer_derivatives, outer_derivatives, 1.0, workspace);
  } catch (const std::invalid_argument&) {
    outer_shape_rejected = true;
  }
  CHECK(outer_shape_rejected);

  teuk::Plus2SpatialRank4View outer_derivatives(
      "plus2_noalloc_valid_outer_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), 2);
  spatial_allocations = 0;
  spatial_deep_copies = 0;
  spatial_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_spatial_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_spatial_deep_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_spatial_fence);
  {
    Kokkos::View<C*> probe("plus2_spatial_callback_probe", 1);
    Kokkos::deep_copy(probe, C{});
    execution.fence("plus2 spatial callback positive control");
  }
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(spatial_allocations > 0);
  CHECK(spatial_deep_copies > 0);
  CHECK(spatial_fences > 0);

  spatial_allocations = 0;
  spatial_deep_copies = 0;
  spatial_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_spatial_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_spatial_deep_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_spatial_fence);
  teuk::evaluate_plus2_spatial_ordered_pairs(
      execution, grid, parameters, angles.cosine, angles.sine,
      input.primitive_value, input.primitive_tangent, input.derivative_value,
      input.derivative_tangent, workspace);
  teuk::evaluate_plus2_spatial_outer_source(
      execution, grid, parameters, angles.cosine, angles.sine,
      workspace.summed_value(), workspace.summed_tangent(), outer_derivatives,
      outer_derivatives, 1.0, workspace);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish plus2 no-allocation evaluation");
  CHECK(spatial_allocations == 0);
  CHECK(spatial_deep_copies == 0);
  CHECK(spatial_fences == 0);
}

}  // namespace
