#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/plus2_source_spatial.hpp"

namespace teuk {

// Narrow production-forcing contract.  J and K require tangents because the
// outer thorn/eth operators contain partial_T.  Q enters algebraically and is
// therefore value-only.  The three Q-only derivative slots intentionally have
// no tangent view.
enum class Plus2ProductionJkDerivative : std::size_t {
  CapitalDelta4Z1 = 0,
  EthPrime4Z1 = 1,
  CapitalDelta2CSharp = 2,
  EthPrime1BSharp = 3,
  CapitalDelta5Z0 = 4,
  Eth5Z0 = 5,
  CapitalDelta2C = 6,
  Eth1B = 7,
  CapitalDelta2V = 8,
  Eth2C = 9,
  EthPrime2CSharp = 10,
  Count = 11,
};

enum class Plus2ProductionQDerivative : std::size_t {
  CapitalDelta2Sig = 0,
  CapitalDelta3Kap = 1,
  EthPrime3Kap = 2,
  Count = 3,
};

enum class Plus2ProductionJkAggregate : std::size_t {
  J = 0,
  K = 1,
  Count = 2,
};

using Plus2ProductionJkDerivativeView = Plus2SpatialRank4View;
using Plus2ProductionQDerivativeView = Plus2SpatialRank4View;
using Plus2ProductionJkAggregateView = Plus2SpatialRank4View;

// Setup owns all output/scratch views.  Evaluation performs no allocation,
// copy, or fence.  Pair-family output is value-only; only target J/K sums
// retain analytic tangents.
class Plus2SourceValueSpatialWorkspace {
 public:
  Plus2SourceValueSpatialWorkspace(
      const ModeRegistry& registry, const std::size_t radial_points,
      const std::size_t theta_points,
      const std::string& label = "plus2_source_value_spatial")
      : mode_count_(registry.size()),
        radial_points_(radial_points),
        theta_points_(theta_points),
        pair_lookup_(label + "_pair_lookup", registry.ordered_pairs().size()),
        target_indices_(label + "_target_indices", registry.targets().size()),
        pair_offsets_(label + "_pair_offsets", registry.targets().size() + 1),
        pair_family_value_(
            label + "_pair_family_value", registry.ordered_pairs().size(),
            static_cast<std::size_t>(Plus2SpatialPairFamily::Count),
            radial_points, theta_points),
        summed_value_(
            label + "_summed_value", registry.size(),
            static_cast<std::size_t>(Plus2SpatialAggregate::Count),
            radial_points, theta_points),
        summed_jk_tangent_(
            label + "_summed_jk_tangent", registry.size(),
            static_cast<std::size_t>(Plus2ProductionJkAggregate::Count),
            radial_points, theta_points),
        source_value_(label + "_source_value", registry.size(), radial_points,
                      theta_points),
        forcing_value_(label + "_forcing_value", registry.size(),
                       radial_points, theta_points) {
    if (!registry.is_closed_under_sharp()) {
      throw std::invalid_argument(
          "spin +2 production source modes must be closed under m -> -m");
    }
    if (radial_points == 0 || theta_points == 0) {
      throw std::invalid_argument(
          "spin +2 production source extents must be nonzero");
    }
    auto host_pairs = Kokkos::create_mirror_view(pair_lookup_);
    auto host_targets = Kokkos::create_mirror_view(target_indices_);
    auto host_offsets = Kokkos::create_mirror_view(pair_offsets_);
    for (std::size_t pair = 0; pair < registry.ordered_pairs().size(); ++pair) {
      host_pairs(pair) =
          make_plus2_pair_lookup(registry, registry.ordered_pairs()[pair]);
    }
    for (std::size_t target = 0; target < registry.targets().size(); ++target) {
      host_targets(target) = registry.index(registry.targets()[target]);
      const auto [begin, end] = registry.pair_range(registry.targets()[target]);
      host_offsets(target) = begin;
      host_offsets(target + 1) = end;
    }
    Kokkos::deep_copy(pair_lookup_, host_pairs);
    Kokkos::deep_copy(target_indices_, host_targets);
    Kokkos::deep_copy(pair_offsets_, host_offsets);
    Kokkos::deep_copy(pair_family_value_, Complex{});
    Kokkos::deep_copy(summed_value_, Complex{});
    Kokkos::deep_copy(summed_jk_tangent_, Complex{});
    Kokkos::deep_copy(source_value_, Complex{});
    Kokkos::deep_copy(forcing_value_, Complex{});
  }

  [[nodiscard]] std::size_t mode_count() const { return mode_count_; }
  [[nodiscard]] std::size_t radial_points() const { return radial_points_; }
  [[nodiscard]] std::size_t theta_points() const { return theta_points_; }
  [[nodiscard]] std::size_t pair_count() const {
    return pair_lookup_.extent(0);
  }
  [[nodiscard]] std::size_t target_count() const {
    return target_indices_.extent(0);
  }
  [[nodiscard]] Plus2SpatialPairLookupView pair_lookup() const {
    return pair_lookup_;
  }
  [[nodiscard]] Plus2SpatialIndexView target_indices() const {
    return target_indices_;
  }
  [[nodiscard]] Plus2SpatialIndexView pair_offsets() const {
    return pair_offsets_;
  }
  [[nodiscard]] Plus2SpatialRank4View pair_family_value() const {
    return pair_family_value_;
  }
  [[nodiscard]] Plus2SpatialRank4View summed_value() const {
    return summed_value_;
  }
  [[nodiscard]] Plus2SpatialRank4View summed_jk_tangent() const {
    return summed_jk_tangent_;
  }
  [[nodiscard]] Plus2SpatialRank3View source_value() const {
    return source_value_;
  }
  [[nodiscard]] Plus2SpatialRank3View forcing_value() const {
    return forcing_value_;
  }

 private:
  std::size_t mode_count_;
  std::size_t radial_points_;
  std::size_t theta_points_;
  Plus2SpatialPairLookupView pair_lookup_;
  Plus2SpatialIndexView target_indices_;
  Plus2SpatialIndexView pair_offsets_;
  Plus2SpatialRank4View pair_family_value_;
  Plus2SpatialRank4View summed_value_;
  Plus2SpatialRank4View summed_jk_tangent_;
  Plus2SpatialRank3View source_value_;
  Plus2SpatialRank3View forcing_value_;
};

namespace detail {

template <class LeftView, class RightView>
bool plus2_same_allocation(const LeftView& left, const RightView& right) {
  return static_cast<const void*>(left.data()) ==
         static_cast<const void*>(right.data());
}

struct Plus2ProductionPairFunctor {
  UniformRadialGrid radial_grid;
  KerrParameters parameters;
  const Real* sin_theta;
  const Real* cos_theta;
  const Complex* primitive_value;
  const Complex* primitive_tangent;
  const Complex* jk_derivative_value;
  const Complex* jk_derivative_tangent;
  const Complex* q_derivative_value;
  const Plus2PairLookup* pair_lookup;
  const std::size_t* target_indices;
  const std::size_t* pair_offsets;
  Complex* pair_value;
  Complex* summed_value;
  Complex* summed_jk_tangent;
  std::size_t radial_points;
  std::size_t theta_points;

  KOKKOS_INLINE_FUNCTION
  Jet1<Complex> primitive(const std::size_t mode,
                          const Plus2SpatialPrimitive component,
                          const std::size_t radial,
                          const std::size_t theta) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
    const std::size_t index = plus2_flat_rank4(
        mode, static_cast<std::size_t>(component), radial, theta, count,
        radial_points, theta_points);
    return {primitive_value[index], primitive_tangent[index]};
  }

  KOKKOS_INLINE_FUNCTION
  Jet1<Complex> jk_derivative(
      const std::size_t mode, const Plus2ProductionJkDerivative component,
      const std::size_t radial, const std::size_t theta) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
    const std::size_t index = plus2_flat_rank4(
        mode, static_cast<std::size_t>(component), radial, theta, count,
        radial_points, theta_points);
    return {jk_derivative_value[index], jk_derivative_tangent[index]};
  }

  KOKKOS_INLINE_FUNCTION
  Complex q_derivative(const std::size_t mode,
                       const Plus2ProductionQDerivative component,
                       const std::size_t radial,
                       const std::size_t theta) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(Plus2ProductionQDerivative::Count);
    return q_derivative_value[plus2_flat_rank4(
        mode, static_cast<std::size_t>(component), radial, theta, count,
        radial_points, theta_points)];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t family_count =
        static_cast<std::size_t>(Plus2SpatialPairFamily::Count);
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t jk_aggregate_count =
        static_cast<std::size_t>(Plus2ProductionJkAggregate::Count);
    const std::size_t plane = radial_points * theta_points;
    const std::size_t target_slot = flat / plane;
    const std::size_t within = flat - target_slot * plane;
    const std::size_t radial = within / theta_points;
    const std::size_t theta = within - radial * theta_points;
    const std::size_t target = target_indices[target_slot];
    const double radius = radial_grid.coordinate(radial);
    const KerrBackgroundPoint background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    Jet1<Complex> summed_J;
    Jet1<Complex> summed_K;
    Complex summed_Q{};

    for (std::size_t pair = pair_offsets[target_slot];
         pair < pair_offsets[target_slot + 1]; ++pair) {
      const Plus2PairLookup lookup = pair_lookup[pair];
      const std::size_t m1 = lookup.index1;
      const std::size_t m2 = lookup.index2;
      const std::size_t sharp1 = lookup.sharp1;
      const Plus2OrderedPairFieldsT<Jet1<Complex>> jet_fields{
          primitive(m1, Plus2SpatialPrimitive::V, radial, theta),
          primitive(m1, Plus2SpatialPrimitive::C, radial, theta),
          jet_conj(
              primitive(sharp1, Plus2SpatialPrimitive::C, radial, theta)),
          primitive(m1, Plus2SpatialPrimitive::B, radial, theta),
          jet_conj(
              primitive(sharp1, Plus2SpatialPrimitive::B, radial, theta)),
          primitive(m1, Plus2SpatialPrimitive::Sig, radial, theta),
          primitive(m1, Plus2SpatialPrimitive::Kap, radial, theta),
          primitive(m1, Plus2SpatialPrimitive::Rh, radial, theta),
          jet_conj(
              primitive(sharp1, Plus2SpatialPrimitive::Rh, radial, theta)),
          primitive(m1, Plus2SpatialPrimitive::Ep, radial, theta),
          jet_conj(
              primitive(sharp1, Plus2SpatialPrimitive::Ep, radial, theta)),
          primitive(m2, Plus2SpatialPrimitive::Z0, radial, theta),
          primitive(m2, Plus2SpatialPrimitive::Z1, radial, theta),
          primitive(m2, Plus2SpatialPrimitive::H, radial, theta),
          primitive(m2, Plus2SpatialPrimitive::Sig, radial, theta),
          primitive(m2, Plus2SpatialPrimitive::Kap, radial, theta)};
      const Plus2OrderedPairJkDerivativesT<Jet1<Complex>> jk_derivatives{
          jk_derivative(m2, Plus2ProductionJkDerivative::CapitalDelta4Z1,
                        radial, theta),
          jk_derivative(m2, Plus2ProductionJkDerivative::EthPrime4Z1,
                        radial, theta),
          jk_derivative(m1,
                        Plus2ProductionJkDerivative::CapitalDelta2CSharp,
                        radial, theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::EthPrime1BSharp,
                        radial, theta),
          jk_derivative(m2, Plus2ProductionJkDerivative::CapitalDelta5Z0,
                        radial, theta),
          jk_derivative(m2, Plus2ProductionJkDerivative::Eth5Z0, radial,
                        theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::CapitalDelta2C,
                        radial, theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::Eth1B, radial,
                        theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::CapitalDelta2V,
                        radial, theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::Eth2C, radial,
                        theta),
          jk_derivative(m1, Plus2ProductionJkDerivative::EthPrime2CSharp,
                        radial, theta)};
      const auto jk = plus2_compact_ordered_pair_jk_source(
          radius, background, jet_fields, jk_derivatives);

      const auto v = [](const Jet1<Complex>& x) { return x.value; };
      const Plus2OrderedPairFieldsT<Complex> value_fields{
          v(jet_fields.V1),       v(jet_fields.C1),
          v(jet_fields.Csharp1),  v(jet_fields.B1),
          v(jet_fields.Bsharp1),  v(jet_fields.Sig1),
          v(jet_fields.Kap1),     v(jet_fields.Rh1),
          v(jet_fields.Rhsharp1), v(jet_fields.Ep1),
          v(jet_fields.Epsharp1), v(jet_fields.Z0_2),
          v(jet_fields.Z1_2),     v(jet_fields.H2),
          v(jet_fields.Sig2),     v(jet_fields.Kap2)};
      const auto q = plus2_compact_ordered_pair_q_source(
          radius, background, value_fields,
          Plus2OrderedPairQDerivativesT<Complex>{
              jk_derivatives.ethprime1_Bsharp1.value,
              q_derivative(m2,
                           Plus2ProductionQDerivative::CapitalDelta2Sig,
                           radial, theta),
              q_derivative(m2,
                           Plus2ProductionQDerivative::CapitalDelta3Kap,
                           radial, theta),
              q_derivative(m2, Plus2ProductionQDerivative::EthPrime3Kap,
                           radial, theta)});
      const Complex family_values[] = {
          jk.C12.total().value, jk.B12.total().value,
          jk.D12.total().value, q.Er12.total(), q.Et12.total(),
          jk.J12.total().value, jk.K12.total().value, q.Q12.total()};
      for (std::size_t family = 0; family < family_count; ++family) {
        pair_value[plus2_flat_rank4(pair, family, radial, theta,
                                    family_count, radial_points,
                                    theta_points)] = family_values[family];
      }
      summed_J += jk.J12.total();
      summed_K += jk.K12.total();
      summed_Q += q.Q12.total();
    }
    const Complex sum_values[] = {summed_J.value, summed_K.value, summed_Q};
    for (std::size_t aggregate = 0; aggregate < aggregate_count;
         ++aggregate) {
      summed_value[plus2_flat_rank4(target, aggregate, radial, theta,
                                    aggregate_count, radial_points,
                                    theta_points)] = sum_values[aggregate];
    }
    const Complex jk_tangents[] = {summed_J.dt, summed_K.dt};
    for (std::size_t aggregate = 0; aggregate < jk_aggregate_count;
         ++aggregate) {
      summed_jk_tangent[plus2_flat_rank4(
          target, aggregate, radial, theta, jk_aggregate_count,
          radial_points, theta_points)] = jk_tangents[aggregate];
    }
  }
};

struct Plus2ProductionOuterFunctor {
  UniformRadialGrid radial_grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* projected_sum_value;
  const Complex* outer_derivative_value;
  const std::size_t* target_indices;
  Complex* source_value;
  Complex* forcing_value;
  double activation;
  std::size_t radial_points;
  std::size_t theta_points;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t derivative_count =
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
    const std::size_t plane = radial_points * theta_points;
    const std::size_t target_slot = flat / plane;
    const std::size_t within = flat - target_slot * plane;
    const std::size_t radial = within / theta_points;
    const std::size_t theta = within - radial * theta_points;
    const std::size_t mode = target_indices[target_slot];
    const double radius = radial_grid.coordinate(radial);
    const auto aggregate = [&](const Plus2SpatialAggregate component) {
      return projected_sum_value[plus2_flat_rank4(
          mode, static_cast<std::size_t>(component), radial, theta,
          aggregate_count, radial_points, theta_points)];
    };
    const auto derivative = [&](const Plus2SpatialOuterDerivative component) {
      return outer_derivative_value[plus2_flat_rank4(
          mode, static_cast<std::size_t>(component), radial, theta,
          derivative_count, radial_points, theta_points)];
    };
    const KerrBackgroundPoint background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    const Complex raw = plus2_compact_outer_source_over_r6(
                            radius, background,
                            aggregate(Plus2SpatialAggregate::J),
                            aggregate(Plus2SpatialAggregate::K),
                            aggregate(Plus2SpatialAggregate::Q),
                            Plus2OuterDerivativesT<Complex>{
                                derivative(
                                    Plus2SpatialOuterDerivative::Thorn5J),
                                derivative(
                                    Plus2SpatialOuterDerivative::Eth6K)})
                            .total();
    const Complex activated = activation * raw;
    const Complex forcing = plus2_coordinate_forcing_from_source_over_r6(
        radius, cos_theta[theta], parameters.spin,
        parameters.compactification_length, activated);
    const std::size_t index = plus2_flat_rank3(
        mode, radial, theta, radial_points, theta_points);
    source_value[index] = activated;
    forcing_value[index] = forcing;
  }
};

static_assert(std::is_trivially_copyable_v<Plus2ProductionPairFunctor>);
static_assert(std::is_trivially_copyable_v<Plus2ProductionOuterFunctor>);
static_assert(sizeof(Plus2ProductionPairFunctor) < 1800);
static_assert(sizeof(Plus2ProductionOuterFunctor) < 1800);

}  // namespace detail

template <class ExecutionSpace>
void evaluate_plus2_production_ordered_pair_values(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const Plus2SpatialThetaView& cos_theta,
    const Plus2SpatialThetaView& sin_theta,
    const Plus2SpatialPrimitiveView& primitive_value,
    const Plus2SpatialPrimitiveView& primitive_tangent,
    const Plus2ProductionJkDerivativeView& jk_derivative_value,
    const Plus2ProductionJkDerivativeView& jk_derivative_tangent,
    const Plus2ProductionQDerivativeView& q_derivative_value,
    Plus2SourceValueSpatialWorkspace& workspace) {
  const std::size_t modes = workspace.mode_count();
  const std::size_t radial_points = workspace.radial_points();
  const std::size_t theta_points = workspace.theta_points();
  constexpr std::size_t primitive_count =
      static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
  constexpr std::size_t jk_derivative_count =
      static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
  constexpr std::size_t q_derivative_count =
      static_cast<std::size_t>(Plus2ProductionQDerivative::Count);
  const bool aliases_input =
      detail::plus2_same_allocation(cos_theta, sin_theta) ||
      detail::plus2_same_allocation(primitive_value, primitive_tangent) ||
      detail::plus2_same_allocation(jk_derivative_value,
                                    jk_derivative_tangent) ||
      detail::plus2_same_allocation(primitive_value, jk_derivative_value) ||
      detail::plus2_same_allocation(primitive_value, q_derivative_value) ||
      detail::plus2_same_allocation(jk_derivative_value,
                                    q_derivative_value);
  const bool aliases_output =
      detail::plus2_same_allocation(primitive_value,
                                    workspace.pair_family_value()) ||
      detail::plus2_same_allocation(primitive_value,
                                    workspace.summed_value()) ||
      detail::plus2_same_allocation(primitive_value,
                                    workspace.summed_jk_tangent()) ||
      detail::plus2_same_allocation(primitive_tangent,
                                    workspace.pair_family_value()) ||
      detail::plus2_same_allocation(primitive_tangent,
                                    workspace.summed_value()) ||
      detail::plus2_same_allocation(primitive_tangent,
                                    workspace.summed_jk_tangent()) ||
      detail::plus2_same_allocation(jk_derivative_value,
                                    workspace.pair_family_value()) ||
      detail::plus2_same_allocation(jk_derivative_value,
                                    workspace.summed_value()) ||
      detail::plus2_same_allocation(jk_derivative_value,
                                    workspace.summed_jk_tangent()) ||
      detail::plus2_same_allocation(jk_derivative_tangent,
                                    workspace.pair_family_value()) ||
      detail::plus2_same_allocation(jk_derivative_tangent,
                                    workspace.summed_value()) ||
      detail::plus2_same_allocation(jk_derivative_tangent,
                                    workspace.summed_jk_tangent()) ||
      detail::plus2_same_allocation(q_derivative_value,
                                    workspace.pair_family_value()) ||
      detail::plus2_same_allocation(q_derivative_value,
                                    workspace.summed_value()) ||
      detail::plus2_same_allocation(q_derivative_value,
                                    workspace.summed_jk_tangent());
  if (radial_grid.size() != radial_points ||
      cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points ||
      !plus2_spatial_rank4_shape(primitive_value, modes, primitive_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(primitive_tangent, modes, primitive_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(jk_derivative_value, modes,
                                 jk_derivative_count, radial_points,
                                 theta_points) ||
      !plus2_spatial_rank4_shape(jk_derivative_tangent, modes,
                                 jk_derivative_count, radial_points,
                                 theta_points) ||
      !plus2_spatial_rank4_shape(q_derivative_value, modes,
                                 q_derivative_count, radial_points,
                                 theta_points) ||
      aliases_input || aliases_output) {
    throw std::invalid_argument(
        "spin +2 production pair extents or aliases are invalid");
  }
  const std::size_t target_points =
      workspace.target_count() * radial_points * theta_points;
  const detail::Plus2ProductionPairFunctor functor{
      radial_grid,
      parameters,
      sin_theta.data(),
      cos_theta.data(),
      primitive_value.data(),
      primitive_tangent.data(),
      jk_derivative_value.data(),
      jk_derivative_tangent.data(),
      q_derivative_value.data(),
      workspace.pair_lookup().data(),
      workspace.target_indices().data(),
      workspace.pair_offsets().data(),
      workspace.pair_family_value().data(),
      workspace.summed_value().data(),
      workspace.summed_jk_tangent().data(),
      radial_points,
      theta_points};
  Kokkos::parallel_for(
      "teuk_plus2_production_ordered_pair_values",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, target_points),
      functor);
}

template <class ExecutionSpace>
void evaluate_plus2_production_outer_source_value(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const Plus2SpatialThetaView& cos_theta,
    const Plus2SpatialThetaView& sin_theta,
    const Plus2SpatialAggregateView& projected_sum_value,
    const Plus2SpatialOuterDerivativeView& outer_derivative_value,
    const double source_activation_multiplier,
    Plus2SourceValueSpatialWorkspace& workspace) {
  const std::size_t modes = workspace.mode_count();
  const std::size_t radial_points = workspace.radial_points();
  const std::size_t theta_points = workspace.theta_points();
  constexpr std::size_t aggregate_count =
      static_cast<std::size_t>(Plus2SpatialAggregate::Count);
  constexpr std::size_t derivative_count =
      static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
  const bool aliases_input =
      detail::plus2_same_allocation(cos_theta, sin_theta) ||
      detail::plus2_same_allocation(projected_sum_value,
                                    outer_derivative_value);
  const bool aliases_output =
      detail::plus2_same_allocation(projected_sum_value,
                                    workspace.source_value()) ||
      detail::plus2_same_allocation(projected_sum_value,
                                    workspace.forcing_value()) ||
      detail::plus2_same_allocation(outer_derivative_value,
                                    workspace.source_value()) ||
      detail::plus2_same_allocation(outer_derivative_value,
                                    workspace.forcing_value());
  if (!std::isfinite(source_activation_multiplier) ||
      radial_grid.size() != radial_points ||
      cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points ||
      !plus2_spatial_rank4_shape(projected_sum_value, modes, aggregate_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(outer_derivative_value, modes,
                                 derivative_count, radial_points,
                                 theta_points) ||
      aliases_input || aliases_output) {
    throw std::invalid_argument(
        "spin +2 production outer extents, aliases, or activation are invalid");
  }
  const std::size_t target_points =
      workspace.target_count() * radial_points * theta_points;
  const detail::Plus2ProductionOuterFunctor functor{
      radial_grid,
      parameters,
      cos_theta.data(),
      sin_theta.data(),
      projected_sum_value.data(),
      outer_derivative_value.data(),
      workspace.target_indices().data(),
      workspace.source_value().data(),
      workspace.forcing_value().data(),
      source_activation_multiplier,
      radial_points,
      theta_points};
  Kokkos::parallel_for(
      "teuk_plus2_production_outer_source_value",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, target_points),
      functor);
}

}  // namespace teuk
