#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_live_source_composition.hpp"
#include "teuk/plus2_source_outer_spatial.hpp"
#include "teuk/plus2_source_primitive_spatial.hpp"
#include "teuk/plus2_source_value_spatial.hpp"

namespace {

using C = teuk::Complex;
using execution_space = teuk::ExecutionSpace;

constexpr std::size_t radial_count = 49;
constexpr std::size_t reconstruction_fields = 5;
constexpr std::uint64_t generation = 71;
constexpr int input_ell_cap = 12;
constexpr int comparison_ell_max = 4;

constexpr std::size_t primitive_count =
    static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count);
constexpr std::size_t jk_count =
    static_cast<std::size_t>(teuk::Plus2ProductionJkDerivative::Count);
constexpr std::size_t q_count =
    static_cast<std::size_t>(teuk::Plus2ProductionQDerivative::Count);
constexpr std::size_t curvature_count = static_cast<std::size_t>(
    teuk::Plus2TransportedCurvatureComponent::Count);
constexpr std::size_t bianchi_count =
    static_cast<std::size_t>(teuk::Plus2BianchiDerivativeComponent::Count);
constexpr std::size_t aggregate_count =
    static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count);
constexpr std::size_t outer_count =
    static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count);
constexpr std::size_t family_count =
    static_cast<std::size_t>(teuk::Plus2SpatialPairFamily::Count);

std::size_t p(const teuk::Plus2SpatialPrimitive value) {
  return static_cast<std::size_t>(value);
}
std::size_t j(const teuk::Plus2ProductionJkDerivative value) {
  return static_cast<std::size_t>(value);
}
std::size_t a(const teuk::Plus2SpatialAggregate value) {
  return static_cast<std::size_t>(value);
}
std::size_t o(const teuk::Plus2SpatialOuterDerivative value) {
  return static_cast<std::size_t>(value);
}
std::size_t f(const teuk::Plus2SpatialPairFamily value) {
  return static_cast<std::size_t>(value);
}

const std::vector<int>& stored_modes() {
  static const std::vector<int> modes{-4, -3, -2, -1, 0, 1, 2, 3, 4};
  return modes;
}
const std::vector<int>& parent_modes() {
  static const std::vector<int> modes{-2, -1, 0, 1, 2};
  return modes;
}
const std::vector<int>& target_modes() {
  static const std::vector<int> modes{-4, -2, 0, 2, 4};
  return modes;
}

double horizon_radius() {
  constexpr double mass = 1.0;
  constexpr double spin = 0.73;
  constexpr double length = 1.2;
  return length * length /
         (mass + std::sqrt(mass * mass - spin * spin));
}

C signed_amplitude(const int m, const std::size_t field,
                   const double amplitude) {
  const double tag = static_cast<double>(field + 1);
  return amplitude *
         C(0.31 + 0.037 * m + 0.013 * tag,
           -0.17 + 0.029 * m - 0.009 * tag + 0.006 * m * m);
}

double radial_profile(const std::size_t field, const double radius) {
  const double tag = static_cast<double>(field + 1);
  return (1.0 + 0.018 * tag * radius + 0.007 * radius * radius) *
         std::exp((0.055 + 0.003 * tag) * radius);
}

C angular_profile(const int spin, const int m, const std::size_t field,
                  const double theta) {
  const int ell_min = std::max(std::abs(spin), std::abs(m));
  C value{};
  double decay = 1.0;
  for (int ell = ell_min; ell <= input_ell_cap; ++ell) {
    const double parity = ((ell + m + static_cast<int>(field)) & 1) ? -1.0 : 1.0;
    const C coefficient =
        decay * C(parity * (1.0 + 0.015 * ell), 0.11 * (ell - ell_min));
    value += coefficient * teuk::angular::spin_weighted_harmonic_theta(
                               ell, m, spin, theta);
    decay *= 0.43;
  }
  return value;
}

struct PrimitiveTarget {
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

struct ModalSample {
  int m = 0;
  int ell = 0;
  std::size_t radial = 0;
  C forcing{};
  C eth6_k{};
  C projected_k{};
  C primitive_kap{};
};

struct AngularRun {
  int ell_max = 0;
  int theta_count = 0;
  std::vector<ModalSample> modal;
  std::array<double, family_count> family_max{};
  double family_closure_max = 0.0;
  double nontarget_max = 0.0;
  double ordered_pair_asymmetry = 0.0;
  double forcing_max = 0.0;
};

template <class HostView>
C analyze_mode(const HostView& values,
               const teuk::angular::GaussLegendreGrid& angular_grid,
               const std::size_t mode, const std::size_t radial,
               const int ell, const int m, const int spin) {
  C coefficient{};
  for (std::size_t node = 0; node < angular_grid.size(); ++node) {
    const double harmonic = teuk::angular::spin_weighted_harmonic_theta(
        ell, m, spin, angular_grid.theta(node));
    coefficient += 2.0 * teuk::angular::pi * angular_grid.weights[node] *
                   harmonic * values(mode, radial, node);
  }
  return coefficient;
}

template <class HostView>
C analyze_field_mode(const HostView& values, const std::size_t field,
                     const teuk::angular::GaussLegendreGrid& angular_grid,
                     const std::size_t mode, const std::size_t radial,
                     const int ell, const int m, const int spin) {
  C coefficient{};
  for (std::size_t node = 0; node < angular_grid.size(); ++node) {
    const double harmonic = teuk::angular::spin_weighted_harmonic_theta(
        ell, m, spin, angular_grid.theta(node));
    coefficient += 2.0 * teuk::angular::pi * angular_grid.weights[node] *
                   harmonic * values(mode, field, radial, node);
  }
  return coefficient;
}

AngularRun run_graph(const int ell_max, const int theta_count,
                     const double amplitude = 1.0) {
  const execution_space execution;
  const teuk::ModeRegistry registry(stored_modes(), parent_modes(),
                                    target_modes());
  const teuk::UniformRadialGrid radial_grid(radial_count, 0.0,
                                            horizon_radius());
  const teuk::KerrParameters parameters{1.0, 0.73, 1.2};
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  teuk::Plus2SpatialThetaView cos_theta("angular_convergence_cos",
                                        theta_count);
  teuk::Plus2SpatialThetaView sin_theta("angular_convergence_sin",
                                        theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (int node = 0; node < theta_count; ++node) {
    host_cos(node) = angular_grid.x[static_cast<std::size_t>(node)];
    host_sin(node) = std::sqrt(1.0 - host_cos(node) * host_cos(node));
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);

  teuk::Plus2PrimitiveConstStageView::non_const_type value(
      "angular_convergence_value", registry.size(), reconstruction_fields,
      radial_count, theta_count);
  teuk::Plus2PrimitiveConstStageView::non_const_type tangent(
      "angular_convergence_tangent", registry.size(), reconstruction_fields,
      radial_count, theta_count);
  teuk::Plus2PrimitiveConstStageView::non_const_type second(
      "angular_convergence_second", registry.size(), reconstruction_fields,
      radial_count, theta_count);
  teuk::Plus2LiveStampView value_stamps(
      "angular_convergence_value_stamps", registry.size(),
      reconstruction_fields, radial_count, theta_count);
  teuk::Plus2LiveStampView tangent_stamps(
      "angular_convergence_tangent_stamps", registry.size(),
      reconstruction_fields, radial_count, theta_count);
  teuk::Plus2LiveStampView second_stamps(
      "angular_convergence_second_stamps", registry.size(),
      reconstruction_fields, radial_count, theta_count);
  teuk::Plus2TransportedCurvatureStorageView curvature(
      "angular_convergence_curvature", registry.size(), curvature_count,
      radial_count, theta_count);
  teuk::Plus2LiveStampView curvature_stamps(
      "angular_convergence_curvature_stamps", registry.size(), curvature_count,
      radial_count, theta_count);
  teuk::Plus2BianchiDerivativeView bianchi(
      "angular_convergence_bianchi", registry.size(), bianchi_count,
      radial_count, theta_count);
  teuk::Plus2LiveStampView bianchi_stamps(
      "angular_convergence_bianchi_stamps", registry.size(), bianchi_count,
      radial_count, theta_count);

  auto hv = Kokkos::create_mirror_view(value);
  auto ht = Kokkos::create_mirror_view(tangent);
  auto hs = Kokkos::create_mirror_view(second);
  auto hc = Kokkos::create_mirror_view(curvature);
  auto hb = Kokkos::create_mirror_view(bianchi);
  constexpr int reconstruction_spins[reconstruction_fields]{0, -1, -2, -1,
                                                             0};
  constexpr int curvature_spins[curvature_count]{2, 1, 2, 1, 2, 1};
  constexpr int bianchi_spins[bianchi_count]{1, 1, 0, 0, 2, 2, 3, 3};
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = radial_grid.coordinate(radial);
      for (int node = 0; node < theta_count; ++node) {
        const double theta = angular_grid.theta(static_cast<std::size_t>(node));
        for (std::size_t field = 0; field < reconstruction_fields; ++field) {
          const C base = signed_amplitude(m, field, amplitude) *
                         radial_profile(field, radius) *
                         angular_profile(reconstruction_spins[field], m, field,
                                         theta);
          const double rate = 0.09 + 0.014 * field - 0.003 * m;
          const double acceleration = -0.021 + 0.002 * field + 0.001 * m;
          hv(mode, field, radial, node) = base;
          ht(mode, field, radial, node) = rate * base;
          hs(mode, field, radial, node) = acceleration * base;
        }
        for (std::size_t field = 0; field < curvature_count; ++field) {
          hc(mode, field, radial, node) =
              0.37 * signed_amplitude(m, field + 11, amplitude) *
              radial_profile(field + 7, radius) *
              angular_profile(curvature_spins[field], m, field + 11, theta);
        }
        for (std::size_t field = 0; field < bianchi_count; ++field) {
          hb(mode, field, radial, node) =
              -0.29 * signed_amplitude(m, field + 23, amplitude) *
              radial_profile(field + 13, radius) *
              angular_profile(bianchi_spins[field], m, field + 23, theta);
        }
      }
    }
  }
  Kokkos::deep_copy(execution, value, hv);
  Kokkos::deep_copy(execution, tangent, ht);
  Kokkos::deep_copy(execution, second, hs);
  Kokkos::deep_copy(execution, curvature, hc);
  Kokkos::deep_copy(execution, bianchi, hb);
  Kokkos::deep_copy(execution, value_stamps, generation);
  Kokkos::deep_copy(execution, tangent_stamps, generation);
  Kokkos::deep_copy(execution, second_stamps, generation);
  Kokkos::deep_copy(execution, curvature_stamps, generation);
  Kokkos::deep_copy(execution, bianchi_stamps, generation);

  teuk::Plus2SpatialPrimitiveView primitive(
      "angular_convergence_primitive", registry.size(), primitive_count,
      radial_count, theta_count);
  teuk::Plus2SpatialPrimitiveView primitive_tangent(
      "angular_convergence_primitive_t", registry.size(), primitive_count,
      radial_count, theta_count);
  teuk::Plus2ProductionJkDerivativeView jk(
      "angular_convergence_jk", registry.size(), jk_count, radial_count,
      theta_count);
  teuk::Plus2ProductionJkDerivativeView jk_tangent(
      "angular_convergence_jk_t", registry.size(), jk_count, radial_count,
      theta_count);
  teuk::Plus2ProductionQDerivativeView q(
      "angular_convergence_q", registry.size(), q_count, radial_count,
      theta_count);
  teuk::Plus2LiveStampView primitive_stamps(
      "angular_convergence_primitive_stamps", registry.size(), primitive_count,
      radial_count, theta_count);
  teuk::Plus2LiveStampView primitive_tangent_stamps(
      "angular_convergence_primitive_t_stamps", registry.size(),
      primitive_count, radial_count, theta_count);
  teuk::Plus2LiveStampView jk_stamps(
      "angular_convergence_jk_stamps", registry.size(), jk_count, radial_count,
      theta_count);
  teuk::Plus2LiveStampView jk_tangent_stamps(
      "angular_convergence_jk_t_stamps", registry.size(), jk_count,
      radial_count, theta_count);
  teuk::Plus2LiveStampView q_stamps(
      "angular_convergence_q_stamps", registry.size(), q_count, radial_count,
      theta_count);
  teuk::Plus2SourcePrimitiveSpatialProducer<execution_space> primitive_producer(
      execution, registry, radial_grid, parameters, ell_max, cos_theta,
      sin_theta, "angular_convergence_primitive_producer",
      teuk::RadialDiscretization::D105);
  primitive_producer.evaluate(
      execution,
      {generation, value, tangent, second, value_stamps, tangent_stamps,
       second_stamps},
      {curvature, curvature_stamps}, {bianchi, bianchi_stamps},
      PrimitiveTarget{generation, primitive, primitive_tangent, jk, jk_tangent,
                      q, primitive_stamps, primitive_tangent_stamps, jk_stamps,
                      jk_tangent_stamps, q_stamps});
  execution.fence("finish angular convergence primitive producer");

  // The live composition owns this typed adapter copy. Reproduce only that
  // ownership transfer here so the test can inspect every intermediate graph
  // quantity without adding production accessors.
  auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, primitive);
  auto hpt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                  primitive_tangent);
  auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, jk);
  auto hjt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                  jk_tangent);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (int node = 0; node < theta_count; ++node) {
        hp(mode, p(teuk::Plus2SpatialPrimitive::Z0), radial, node) =
            hc(mode, static_cast<std::size_t>(
                         teuk::Plus2TransportedCurvatureComponent::Z0),
               radial, node);
        hp(mode, p(teuk::Plus2SpatialPrimitive::Z1), radial, node) =
            hc(mode, static_cast<std::size_t>(
                         teuk::Plus2TransportedCurvatureComponent::Z1),
               radial, node);
        hpt(mode, p(teuk::Plus2SpatialPrimitive::Z0), radial, node) =
            hc(mode, static_cast<std::size_t>(
                         teuk::Plus2TransportedCurvatureComponent::Z0T),
               radial, node);
        hpt(mode, p(teuk::Plus2SpatialPrimitive::Z1), radial, node) =
            hc(mode, static_cast<std::size_t>(
                         teuk::Plus2TransportedCurvatureComponent::Z1T),
               radial, node);
        constexpr teuk::Plus2ProductionJkDerivative out[4]{
            teuk::Plus2ProductionJkDerivative::CapitalDelta4Z1,
            teuk::Plus2ProductionJkDerivative::EthPrime4Z1,
            teuk::Plus2ProductionJkDerivative::CapitalDelta5Z0,
            teuk::Plus2ProductionJkDerivative::Eth5Z0};
        constexpr teuk::Plus2BianchiDerivativeComponent in[4]{
            teuk::Plus2BianchiDerivativeComponent::CapitalDelta4Z1,
            teuk::Plus2BianchiDerivativeComponent::EthPrime4Z1,
            teuk::Plus2BianchiDerivativeComponent::CapitalDelta5Z0,
            teuk::Plus2BianchiDerivativeComponent::Eth5Z0};
        constexpr teuk::Plus2BianchiDerivativeComponent in_t[4]{
            teuk::Plus2BianchiDerivativeComponent::CapitalDelta4Z1T,
            teuk::Plus2BianchiDerivativeComponent::EthPrime4Z1T,
            teuk::Plus2BianchiDerivativeComponent::CapitalDelta5Z0T,
            teuk::Plus2BianchiDerivativeComponent::Eth5Z0T};
        for (std::size_t slot = 0; slot < 4; ++slot) {
          hj(mode, j(out[slot]), radial, node) =
              hb(mode, static_cast<std::size_t>(in[slot]), radial, node);
          hjt(mode, j(out[slot]), radial, node) =
              hb(mode, static_cast<std::size_t>(in_t[slot]), radial, node);
        }
      }
    }
  }
  Kokkos::deep_copy(execution, primitive, hp);
  Kokkos::deep_copy(execution, primitive_tangent, hpt);
  Kokkos::deep_copy(execution, jk, hj);
  Kokkos::deep_copy(execution, jk_tangent, hjt);

  teuk::Plus2SourceValueSpatialWorkspace source(
      registry, radial_count, theta_count, "angular_convergence_source");
  teuk::evaluate_plus2_production_ordered_pair_values(
      execution, radial_grid, parameters, cos_theta, sin_theta, primitive,
      primitive_tangent, jk, jk_tangent, q, source);

  teuk::Plus2SpatialAggregateView projected(
      "angular_convergence_projected", registry.size(), aggregate_count,
      radial_count, theta_count);
  teuk::Plus2SpatialOuterDerivativeView outer(
      "angular_convergence_outer", registry.size(), outer_count, radial_count,
      theta_count);
  teuk::Plus2LiveStampView projected_stamps(
      "angular_convergence_projected_stamps", registry.size(), aggregate_count,
      radial_count, theta_count);
  teuk::Plus2LiveStampView outer_stamps(
      "angular_convergence_outer_stamps", registry.size(), outer_count,
      radial_count, theta_count);
  teuk::Plus2SourceOuterSpatialProducer<execution_space> outer_producer(
      execution, registry, radial_grid, parameters, ell_max, cos_theta,
      sin_theta, "angular_convergence_outer_producer",
      teuk::RadialDiscretization::D105);
  outer_producer.evaluate(
      execution,
      {generation, source.summed_value(), source.summed_jk_tangent()},
      teuk::Plus2LiveOuterWriteTarget{generation, projected, outer,
                                      projected_stamps, outer_stamps});
  teuk::evaluate_plus2_production_outer_source_value(
      execution, radial_grid, parameters, cos_theta, sin_theta, projected,
      outer, 1.0, source);
  execution.fence("finish complete angular convergence graph");

  const auto hpair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, source.pair_family_value());
  const auto hsum = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, source.summed_value());
  const auto hprojected = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, projected);
  const auto houter =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outer);
  const auto hforcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, source.forcing_value());

  AngularRun result;
  result.ell_max = ell_max;
  result.theta_count = theta_count;
  for (std::size_t pair = 0; pair < registry.ordered_pairs().size(); ++pair) {
    for (std::size_t field = 0; field < family_count; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (int node = 0; node < theta_count; ++node) {
          result.family_max[field] =
              std::max(result.family_max[field],
                       Kokkos::abs(hpair(pair, field, radial, node)));
        }
      }
    }
  }
  for (const int target : target_modes()) {
    const auto [begin, end] = registry.pair_range(target);
    const std::size_t mode = registry.index(target);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (int node = 0; node < theta_count; ++node) {
        C sum_j{};
        C sum_k{};
        C sum_q{};
        for (std::size_t pair = begin; pair < end; ++pair) {
          sum_j += hpair(pair, f(teuk::Plus2SpatialPairFamily::J), radial,
                         node);
          sum_k += hpair(pair, f(teuk::Plus2SpatialPairFamily::K), radial,
                         node);
          sum_q += hpair(pair, f(teuk::Plus2SpatialPairFamily::Q), radial,
                         node);
        }
        result.family_closure_max = std::max(
            {result.family_closure_max,
             Kokkos::abs(sum_j - hsum(mode, a(teuk::Plus2SpatialAggregate::J),
                                      radial, node)),
             Kokkos::abs(sum_k - hsum(mode, a(teuk::Plus2SpatialAggregate::K),
                                      radial, node)),
             Kokkos::abs(sum_q - hsum(mode, a(teuk::Plus2SpatialAggregate::Q),
                                      radial, node))});
        result.forcing_max =
            std::max(result.forcing_max,
                     Kokkos::abs(hforcing(mode, radial, node)));
      }
    }
  }
  for (const int m : stored_modes()) {
    if (registry.is_target(m)) continue;
    const std::size_t mode = registry.index(m);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (int node = 0; node < theta_count; ++node) {
        result.nontarget_max = std::max(
            {result.nontarget_max, Kokkos::abs(hforcing(mode, radial, node)),
             Kokkos::abs(hsum(mode, a(teuk::Plus2SpatialAggregate::J), radial,
                              node)),
             Kokkos::abs(hsum(mode, a(teuk::Plus2SpatialAggregate::K), radial,
                              node)),
             Kokkos::abs(hsum(mode, a(teuk::Plus2SpatialAggregate::Q), radial,
                              node)),
             Kokkos::abs(hprojected(
                 mode, a(teuk::Plus2SpatialAggregate::K), radial, node)),
             Kokkos::abs(houter(
                 mode, o(teuk::Plus2SpatialOuterDerivative::Eth6K), radial,
                 node))});
      }
    }
  }
  const auto [zero_begin, zero_end] = registry.pair_range(0);
  for (std::size_t left = zero_begin; left < zero_end; ++left) {
    for (std::size_t right = left + 1; right < zero_end; ++right) {
      const auto& lhs = registry.ordered_pairs()[left];
      const auto& rhs = registry.ordered_pairs()[right];
      if (lhs.m1 == rhs.m2 && lhs.m2 == rhs.m1 && lhs.m1 != lhs.m2) {
        result.ordered_pair_asymmetry = std::max(
            result.ordered_pair_asymmetry,
            Kokkos::abs(hpair(left, f(teuk::Plus2SpatialPairFamily::J),
                              radial_count / 2, theta_count / 2) -
                        hpair(right, f(teuk::Plus2SpatialPairFamily::J),
                              radial_count / 2, theta_count / 2)));
      }
    }
  }

  for (const int m : target_modes()) {
    const std::size_t mode = registry.index(m);
    const int ell_min = std::max(2, std::abs(m));
    for (int ell = ell_min; ell <= comparison_ell_max; ++ell) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        result.modal.push_back(
            {m,
             ell,
             radial,
             analyze_mode(hforcing, angular_grid, mode, radial, ell, m, 2),
             analyze_field_mode(
                 houter,
                 o(teuk::Plus2SpatialOuterDerivative::Eth6K), angular_grid,
                 mode, radial, ell, m, 2),
             analyze_field_mode(hprojected,
                                a(teuk::Plus2SpatialAggregate::K),
                                angular_grid, mode, radial, ell, m, 1),
             analyze_field_mode(hp, p(teuk::Plus2SpatialPrimitive::Kap),
                                angular_grid, mode, radial, ell, m, 1)});
      }
    }
  }
  return result;
}

struct DifferenceNorm {
  double rms = 0.0;
  double maximum = 0.0;
  double endpoint = 0.0;
};

template <class Member>
DifferenceNorm difference_norm(const AngularRun& trial,
                               const AngularRun& reference, Member member) {
  CHECK(trial.modal.size() == reference.modal.size());
  double error2 = 0.0;
  DifferenceNorm result;
  for (std::size_t i = 0; i < trial.modal.size(); ++i) {
    CHECK(trial.modal[i].m == reference.modal[i].m);
    CHECK(trial.modal[i].ell == reference.modal[i].ell);
    CHECK(trial.modal[i].radial == reference.modal[i].radial);
    const double error = Kokkos::abs(member(trial.modal[i]) -
                                     member(reference.modal[i]));
    error2 += error * error;
    result.maximum = std::max(result.maximum, error);
    if (trial.modal[i].radial == 0 ||
        trial.modal[i].radial + 1 == radial_count) {
      result.endpoint = std::max(result.endpoint, error);
    }
  }
  result.rms = std::sqrt(error2 / static_cast<double>(trial.modal.size()));
  return result;
}

double relative_difference(const C left, const C right) {
  return Kokkos::abs(left - right) /
         std::max({1.0, Kokkos::abs(left), Kokkos::abs(right)});
}

}  // namespace

TEST_CASE("complete concrete plus2 source graph closes signed ordered families and scales quadratically") {
  const AngularRun base = run_graph(10, 30, 1.0);
  constexpr double scale = -1.7;
  const AngularRun scaled = run_graph(10, 30, scale);
  for (const double maximum : base.family_max) CHECK(maximum > 1.0e-9);
  CHECK(base.family_closure_max < 2.0e-12);
  CHECK(base.nontarget_max < 2.0e-13);
  CHECK(base.ordered_pair_asymmetry > 1.0e-8);
  CHECK(base.forcing_max > 1.0e-8);
  CHECK(base.modal.size() == scaled.modal.size());
  double scaling_error = 0.0;
  for (std::size_t i = 0; i < base.modal.size(); ++i) {
    scaling_error = std::max(
        scaling_error,
        relative_difference(scaled.modal[i].forcing,
                            scale * scale * base.modal[i].forcing));
  }
  CHECK(scaling_error < 2.0e-11);
}

TEST_CASE("complete concrete plus2 source graph separates Galerkin and quadrature convergence") {
  const AngularRun reference = run_graph(12, 36);
  const AngularRun band4 = run_graph(4, 36);
  const AngularRun band6 = run_graph(6, 36);
  const AngularRun band8 = run_graph(8, 36);
  const AngularRun band10 = run_graph(10, 36);
  const auto forcing = [](const ModalSample& x) { return x.forcing; };
  const auto eth6 = [](const ModalSample& x) { return x.eth6_k; };
  const auto projected_k = [](const ModalSample& x) { return x.projected_k; };
  const auto kap = [](const ModalSample& x) { return x.primitive_kap; };
  const DifferenceNorm b4 = difference_norm(band4, reference, forcing);
  const DifferenceNorm b6 = difference_norm(band6, reference, forcing);
  const DifferenceNorm b8 = difference_norm(band8, reference, forcing);
  const DifferenceNorm b10 = difference_norm(band10, reference, forcing);
  const DifferenceNorm e4 = difference_norm(band4, reference, eth6);
  const DifferenceNorm e6 = difference_norm(band6, reference, eth6);
  const DifferenceNorm e8 = difference_norm(band8, reference, eth6);
  const DifferenceNorm e10 = difference_norm(band10, reference, eth6);
  const DifferenceNorm p4 = difference_norm(band4, reference, projected_k);
  const DifferenceNorm p6 = difference_norm(band6, reference, projected_k);
  const DifferenceNorm p8 = difference_norm(band8, reference, projected_k);
  const DifferenceNorm p10 = difference_norm(band10, reference, projected_k);
  const DifferenceNorm k4 = difference_norm(band4, reference, kap);
  const DifferenceNorm k6 = difference_norm(band6, reference, kap);
  const DifferenceNorm k8 = difference_norm(band8, reference, kap);
  const DifferenceNorm k10 = difference_norm(band10, reference, kap);
  std::cout << "plus2 complete angular band forcing rms " << b4.rms << " "
            << b6.rms << " " << b8.rms << " " << b10.rms
            << " max " << b4.maximum << " " << b6.maximum << " "
            << b8.maximum << " " << b10.maximum << " endpoint "
            << b4.endpoint << " " << b6.endpoint << " " << b8.endpoint
            << " " << b10.endpoint << '\n';
  std::cout << "plus2 complete angular band eth6 rms " << e4.rms << " "
            << e6.rms << " " << e8.rms << " " << e10.rms << '\n';
  std::cout << "plus2 complete angular band projected K rms " << p4.rms << " "
            << p6.rms << " " << p8.rms << " " << p10.rms << '\n';
  std::cout << "plus2 complete angular band kap rms " << k4.rms << " "
            << k6.rms << " " << k8.rms << " " << k10.rms << '\n';
  CHECK(b4.rms > b6.rms);
  CHECK(b6.rms > b8.rms);
  CHECK(b8.rms > b10.rms);
  CHECK(b4.maximum > b6.maximum);
  CHECK(b6.maximum > b8.maximum);
  CHECK(b8.maximum > b10.maximum);
  CHECK(b4.endpoint > b6.endpoint);
  CHECK(b6.endpoint > b8.endpoint);
  CHECK(b8.endpoint > b10.endpoint);
  CHECK(b4.rms / b10.rms > 20.0);
  CHECK(e4.rms / e10.rms > 20.0);
  CHECK(p4.rms / p10.rms > 20.0);
  CHECK(k4.rms / k10.rms > 20.0);
  CHECK(b10.rms < 1.0e-5);
  CHECK(b10.maximum < 6.0e-5);
  CHECK(b10.endpoint < 6.0e-5);
  CHECK(e10.rms < 1.0e-5);
  CHECK(p10.rms < 1.0e-5);
  CHECK(k10.rms < 4.0e-9);

  const AngularRun nodes12 = run_graph(10, 12);
  const AngularRun nodes18 = run_graph(10, 18);
  const AngularRun nodes24 = run_graph(10, 24);
  const DifferenceNorm n12 = difference_norm(nodes12, band10, forcing);
  const DifferenceNorm n18 = difference_norm(nodes18, band10, forcing);
  const DifferenceNorm n24 = difference_norm(nodes24, band10, forcing);
  std::cout << "plus2 complete angular quadrature forcing rms " << n12.rms
            << " " << n18.rms << " " << n24.rms << " max "
            << n12.maximum << " " << n18.maximum << " " << n24.maximum
            << " endpoint " << n12.endpoint << " " << n18.endpoint << " "
            << n24.endpoint << '\n';
  CHECK(n12.rms > n18.rms);
  CHECK(n18.rms > n24.rms);
  CHECK(n12.maximum > n18.maximum);
  CHECK(n18.maximum > n24.maximum);
  CHECK(n12.endpoint > n18.endpoint);
  CHECK(n18.endpoint > n24.endpoint);
  CHECK(n12.rms / n24.rms > 20.0);
  CHECK(n24.rms < 2.0e-14);
  CHECK(n24.maximum < 2.0e-14);
  CHECK(n24.endpoint < 2.0e-14);
}
