#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/jet.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_source.hpp"
#include "teuk/plus2_source_primitives.hpp"
#include "teuk/types.hpp"

namespace teuk {

// The fourteen rows of PLUS2_SOURCE_INPUT_MANIFEST.csv.  Inputs are already
// qualified regular primitives; this layer never constructs curvature
// quotients or applies radial/angular operators.
enum class Plus2SpatialPrimitive : std::size_t {
  Z0 = 0,
  Z1 = 1,
  H = 2,
  Sig = 3,
  Kap = 4,
  Rh = 5,
  Ta = 6,
  Al = 7,
  Be = 8,
  Ep = 9,
  Pi = 10,
  V = 11,
  C = 12,
  B = 13,
  Count = 14,
};

// Explicit stage-local slots in Plus2OrderedPairDerivativesT.  A component
// carrying "sharp" is already the qualified derivative for that signed mode;
// this evaluator does not manufacture an operator by conjugation.
enum class Plus2SpatialPairDerivative : std::size_t {
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
  CapitalDelta2Sig = 11,
  CapitalDelta3Kap = 12,
  EthPrime3Kap = 13,
  Count = 14,
};

enum class Plus2SpatialPairFamily : std::size_t {
  C = 0,
  B = 1,
  D = 2,
  Er = 3,
  Et = 4,
  J = 5,
  K = 6,
  Q = 7,
  Count = 8,
};

enum class Plus2SpatialAggregate : std::size_t {
  J = 0,
  K = 1,
  Q = 2,
  Count = 3,
};

enum class Plus2SpatialOuterDerivative : std::size_t {
  Thorn5J = 0,
  Eth6K = 1,
  Count = 2,
};

using Plus2SpatialRank4View =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2SpatialRank3View =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using Plus2SpatialPrimitiveView = Plus2SpatialRank4View;
using Plus2SpatialPairDerivativeView = Plus2SpatialRank4View;
using Plus2SpatialAggregateView = Plus2SpatialRank4View;
using Plus2SpatialOuterDerivativeView = Plus2SpatialRank4View;
using Plus2SpatialThetaView = Kokkos::View<Real*, MemorySpace>;
using Plus2SpatialPairLookupView =
    Kokkos::View<Plus2PairLookup*, MemorySpace>;
using Plus2SpatialIndexView = Kokkos::View<std::size_t*, MemorySpace>;

// Allocation-owning setup object.  Evaluation writes every pair and target
// entry in place.  Non-target mode slots are initialized to zero once here and
// remain zero because both kernels launch only over configured targets.
class Plus2SourceSpatialWorkspace {
 public:
  Plus2SourceSpatialWorkspace(
      const ModeRegistry& registry, const std::size_t radial_points,
      const std::size_t theta_points,
      const std::string& label = "plus2_source_spatial")
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
        pair_family_tangent_(
            label + "_pair_family_tangent", registry.ordered_pairs().size(),
            static_cast<std::size_t>(Plus2SpatialPairFamily::Count),
            radial_points, theta_points),
        summed_value_(
            label + "_summed_value", registry.size(),
            static_cast<std::size_t>(Plus2SpatialAggregate::Count),
            radial_points, theta_points),
        summed_tangent_(
            label + "_summed_tangent", registry.size(),
            static_cast<std::size_t>(Plus2SpatialAggregate::Count),
            radial_points, theta_points),
        source_value_(label + "_source_value", registry.size(), radial_points,
                      theta_points),
        source_tangent_(label + "_source_tangent", registry.size(),
                        radial_points, theta_points),
        forcing_value_(label + "_forcing_value", registry.size(),
                       radial_points, theta_points),
        forcing_tangent_(label + "_forcing_tangent", registry.size(),
                         radial_points, theta_points) {
    if (!registry.is_closed_under_sharp()) {
      throw std::invalid_argument(
          "spin +2 spatial source modes must be closed under m -> -m");
    }
    if (radial_points == 0 || theta_points == 0) {
      throw std::invalid_argument(
          "spin +2 spatial source extents must be nonzero");
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
    Kokkos::deep_copy(pair_family_tangent_, Complex{});
    Kokkos::deep_copy(summed_value_, Complex{});
    Kokkos::deep_copy(summed_tangent_, Complex{});
    Kokkos::deep_copy(source_value_, Complex{});
    Kokkos::deep_copy(source_tangent_, Complex{});
    Kokkos::deep_copy(forcing_value_, Complex{});
    Kokkos::deep_copy(forcing_tangent_, Complex{});
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
  [[nodiscard]] Plus2SpatialRank4View pair_family_tangent() const {
    return pair_family_tangent_;
  }
  [[nodiscard]] Plus2SpatialRank4View summed_value() const {
    return summed_value_;
  }
  [[nodiscard]] Plus2SpatialRank4View summed_tangent() const {
    return summed_tangent_;
  }
  [[nodiscard]] Plus2SpatialRank3View source_value() const {
    return source_value_;
  }
  [[nodiscard]] Plus2SpatialRank3View source_tangent() const {
    return source_tangent_;
  }
  [[nodiscard]] Plus2SpatialRank3View forcing_value() const {
    return forcing_value_;
  }
  [[nodiscard]] Plus2SpatialRank3View forcing_tangent() const {
    return forcing_tangent_;
  }

 private:
  std::size_t mode_count_;
  std::size_t radial_points_;
  std::size_t theta_points_;
  Plus2SpatialPairLookupView pair_lookup_;
  Plus2SpatialIndexView target_indices_;
  Plus2SpatialIndexView pair_offsets_;
  Plus2SpatialRank4View pair_family_value_;
  Plus2SpatialRank4View pair_family_tangent_;
  Plus2SpatialRank4View summed_value_;
  Plus2SpatialRank4View summed_tangent_;
  Plus2SpatialRank3View source_value_;
  Plus2SpatialRank3View source_tangent_;
  Plus2SpatialRank3View forcing_value_;
  Plus2SpatialRank3View forcing_tangent_;
};

inline bool plus2_spatial_rank4_shape(
    const Plus2SpatialRank4View& view, const std::size_t modes,
    const std::size_t components, const std::size_t radial,
    const std::size_t theta) {
  return view.extent(0) == modes && view.extent(1) == components &&
         view.extent(2) == radial && view.extent(3) == theta;
}

namespace detail {

KOKKOS_INLINE_FUNCTION
std::size_t plus2_flat_rank4(const std::size_t i0, const std::size_t i1,
                             const std::size_t i2, const std::size_t i3,
                             const std::size_t extent1,
                             const std::size_t extent2,
                             const std::size_t extent3) {
  return ((i0 * extent1 + i1) * extent2 + i2) * extent3 + i3;
}

KOKKOS_INLINE_FUNCTION
std::size_t plus2_flat_rank3(const std::size_t i0, const std::size_t i1,
                             const std::size_t i2,
                             const std::size_t extent1,
                             const std::size_t extent2) {
  return (i0 * extent1 + i1) * extent2 + i2;
}

// Raw pointers keep the SYCL kernel functor small and trivially copyable.
// Capturing all rank-4 Views directly exceeds Kokkos' direct-kernel-argument
// threshold on Level Zero, which makes the backend use reusable indirect
// kernel storage and fence before overwriting it on the next launch.
struct Plus2SpatialPairFunctor {
  UniformRadialGrid radial_grid;
  KerrParameters parameters;
  const Real* sin_theta;
  const Real* cos_theta;
  const Complex* primitive_value;
  const Complex* primitive_tangent;
  const Complex* derivative_value;
  const Complex* derivative_tangent;
  const Plus2PairLookup* pair_lookup;
  const std::size_t* target_indices;
  const std::size_t* pair_offsets;
  Complex* pair_value;
  Complex* pair_tangent;
  Complex* summed_value;
  Complex* summed_tangent;
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
  Jet1<Complex> derivative(const std::size_t mode,
                           const Plus2SpatialPairDerivative component,
                           const std::size_t radial,
                           const std::size_t theta) const {
    constexpr std::size_t count =
        static_cast<std::size_t>(Plus2SpatialPairDerivative::Count);
    const std::size_t index = plus2_flat_rank4(
        mode, static_cast<std::size_t>(component), radial, theta, count,
        radial_points, theta_points);
    return {derivative_value[index], derivative_tangent[index]};
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    constexpr std::size_t family_count =
        static_cast<std::size_t>(Plus2SpatialPairFamily::Count);
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
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
    Jet1<Complex> summed_Q;

    for (std::size_t pair = pair_offsets[target_slot];
         pair < pair_offsets[target_slot + 1]; ++pair) {
      const Plus2PairLookup lookup = pair_lookup[pair];
      const std::size_t m1 = lookup.index1;
      const std::size_t m2 = lookup.index2;
      const std::size_t sharp1 = lookup.sharp1;
      const Plus2OrderedPairFieldsT<Jet1<Complex>> fields{
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
      const Plus2OrderedPairDerivativesT<Jet1<Complex>> derivatives{
          derivative(m2, Plus2SpatialPairDerivative::CapitalDelta4Z1,
                     radial, theta),
          derivative(m2, Plus2SpatialPairDerivative::EthPrime4Z1, radial,
                     theta),
          derivative(m1, Plus2SpatialPairDerivative::CapitalDelta2CSharp,
                     radial, theta),
          derivative(m1, Plus2SpatialPairDerivative::EthPrime1BSharp, radial,
                     theta),
          derivative(m2, Plus2SpatialPairDerivative::CapitalDelta5Z0,
                     radial, theta),
          derivative(m2, Plus2SpatialPairDerivative::Eth5Z0, radial, theta),
          derivative(m1, Plus2SpatialPairDerivative::CapitalDelta2C, radial,
                     theta),
          derivative(m1, Plus2SpatialPairDerivative::Eth1B, radial, theta),
          derivative(m1, Plus2SpatialPairDerivative::CapitalDelta2V, radial,
                     theta),
          derivative(m1, Plus2SpatialPairDerivative::Eth2C, radial, theta),
          derivative(m1, Plus2SpatialPairDerivative::EthPrime2CSharp, radial,
                     theta),
          derivative(m2, Plus2SpatialPairDerivative::CapitalDelta2Sig,
                     radial, theta),
          derivative(m2, Plus2SpatialPairDerivative::CapitalDelta3Kap,
                     radial, theta),
          derivative(m2, Plus2SpatialPairDerivative::EthPrime3Kap, radial,
                     theta)};
      const auto source = plus2_compact_ordered_pair_source(
          radius, background, fields, derivatives);
      const Jet1<Complex> families[] = {
          source.C12.total(), source.B12.total(), source.D12.total(),
          source.Er12.total(), source.Et12.total(), source.J12.total(),
          source.K12.total(), source.Q12.total()};
      for (std::size_t family = 0; family < family_count; ++family) {
        const std::size_t index = plus2_flat_rank4(
            pair, family, radial, theta, family_count, radial_points,
            theta_points);
        pair_value[index] = families[family].value;
        pair_tangent[index] = families[family].dt;
      }
      summed_J +=
          families[static_cast<std::size_t>(Plus2SpatialPairFamily::J)];
      summed_K +=
          families[static_cast<std::size_t>(Plus2SpatialPairFamily::K)];
      summed_Q +=
          families[static_cast<std::size_t>(Plus2SpatialPairFamily::Q)];
    }
    const Jet1<Complex> sums[] = {summed_J, summed_K, summed_Q};
    for (std::size_t aggregate = 0; aggregate < aggregate_count;
         ++aggregate) {
      const std::size_t index = plus2_flat_rank4(
          target, aggregate, radial, theta, aggregate_count, radial_points,
          theta_points);
      summed_value[index] = sums[aggregate].value;
      summed_tangent[index] = sums[aggregate].dt;
    }
  }
};

struct Plus2SpatialOuterFunctor {
  UniformRadialGrid radial_grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const Complex* projected_sum_value;
  const Complex* projected_sum_tangent;
  const Complex* outer_derivative_value;
  const Complex* outer_derivative_tangent;
  const std::size_t* target_indices;
  Complex* source_value;
  Complex* source_tangent;
  Complex* forcing_value;
  Complex* forcing_tangent;
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
      const std::size_t index = plus2_flat_rank4(
          mode, static_cast<std::size_t>(component), radial, theta,
          aggregate_count, radial_points, theta_points);
      return Jet1<Complex>{projected_sum_value[index],
                           projected_sum_tangent[index]};
    };
    const auto derivative = [&](const Plus2SpatialOuterDerivative component) {
      const std::size_t index = plus2_flat_rank4(
          mode, static_cast<std::size_t>(component), radial, theta,
          derivative_count, radial_points, theta_points);
      return Jet1<Complex>{outer_derivative_value[index],
                           outer_derivative_tangent[index]};
    };
    const KerrBackgroundPoint background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    const auto raw = plus2_compact_outer_source_over_r6(
        radius, background, aggregate(Plus2SpatialAggregate::J),
        aggregate(Plus2SpatialAggregate::K),
        aggregate(Plus2SpatialAggregate::Q),
        Plus2OuterDerivativesT<Jet1<Complex>>{
            derivative(Plus2SpatialOuterDerivative::Thorn5J),
            derivative(Plus2SpatialOuterDerivative::Eth6K)});
    const Jet1<Complex> activated = activation * raw.total();
    const Jet1<Complex> forcing =
        plus2_coordinate_forcing_from_source_over_r6(
            radius, cos_theta[theta], parameters.spin,
            parameters.compactification_length, activated);
    const std::size_t index = plus2_flat_rank3(
        mode, radial, theta, radial_points, theta_points);
    source_value[index] = activated.value;
    source_tangent[index] = activated.dt;
    forcing_value[index] = forcing.value;
    forcing_tangent[index] = forcing.dt;
  }
};

static_assert(std::is_trivially_copyable_v<Plus2SpatialPairFunctor>);
static_assert(std::is_trivially_copyable_v<Plus2SpatialOuterFunctor>);
static_assert(sizeof(Plus2SpatialPairFunctor) < 1800);
static_assert(sizeof(Plus2SpatialOuterFunctor) < 1800);

}  // namespace detail

template <class ExecutionSpace>
void evaluate_plus2_spatial_ordered_pairs(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const Plus2SpatialThetaView& cos_theta,
    const Plus2SpatialThetaView& sin_theta,
    const Plus2SpatialPrimitiveView& primitive_value,
    const Plus2SpatialPrimitiveView& primitive_tangent,
    const Plus2SpatialPairDerivativeView& derivative_value,
    const Plus2SpatialPairDerivativeView& derivative_tangent,
    Plus2SourceSpatialWorkspace& workspace) {
  const std::size_t modes = workspace.mode_count();
  const std::size_t radial_points = workspace.radial_points();
  const std::size_t theta_points = workspace.theta_points();
  constexpr std::size_t primitive_count =
      static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
  constexpr std::size_t derivative_count =
      static_cast<std::size_t>(Plus2SpatialPairDerivative::Count);
  if (radial_grid.size() != radial_points ||
      cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points ||
      !plus2_spatial_rank4_shape(primitive_value, modes, primitive_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(primitive_tangent, modes, primitive_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(derivative_value, modes, derivative_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(derivative_tangent, modes, derivative_count,
                                 radial_points, theta_points)) {
    throw std::invalid_argument(
        "spin +2 spatial pair input extents do not match workspace");
  }

  const auto pair_lookup = workspace.pair_lookup();
  const auto target_indices = workspace.target_indices();
  const auto pair_offsets = workspace.pair_offsets();
  const auto pair_value = workspace.pair_family_value();
  const auto pair_tangent = workspace.pair_family_tangent();
  const auto summed_value = workspace.summed_value();
  const auto summed_tangent = workspace.summed_tangent();
  const std::size_t target_points =
      workspace.target_count() * radial_points * theta_points;
  const detail::Plus2SpatialPairFunctor functor{
      radial_grid,
      parameters,
      sin_theta.data(),
      cos_theta.data(),
      primitive_value.data(),
      primitive_tangent.data(),
      derivative_value.data(),
      derivative_tangent.data(),
      pair_lookup.data(),
      target_indices.data(),
      pair_offsets.data(),
      pair_value.data(),
      pair_tangent.data(),
      summed_value.data(),
      summed_tangent.data(),
      radial_points,
      theta_points};
  Kokkos::parallel_for("teuk_plus2_spatial_ordered_pairs",
                       Kokkos::RangePolicy<ExecutionSpace>(execution, 0,
                                                           target_points),
                       functor);
}

template <class ExecutionSpace>
void evaluate_plus2_spatial_outer_source(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const Plus2SpatialThetaView& cos_theta,
    const Plus2SpatialThetaView& sin_theta,
    const Plus2SpatialAggregateView& projected_sum_value,
    const Plus2SpatialAggregateView& projected_sum_tangent,
    const Plus2SpatialOuterDerivativeView& outer_derivative_value,
    const Plus2SpatialOuterDerivativeView& outer_derivative_tangent,
    const double source_activation_multiplier,
    Plus2SourceSpatialWorkspace& workspace) {
  // Activation is applied exactly once at the final outer source.  Pair
  // families and projected J/K/Q therefore remain raw diagnostics and must
  // not be pre-multiplied by callers before entering this function.
  const std::size_t modes = workspace.mode_count();
  const std::size_t radial_points = workspace.radial_points();
  const std::size_t theta_points = workspace.theta_points();
  constexpr std::size_t aggregate_count =
      static_cast<std::size_t>(Plus2SpatialAggregate::Count);
  constexpr std::size_t derivative_count =
      static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
  if (!std::isfinite(source_activation_multiplier) ||
      radial_grid.size() != radial_points ||
      cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points ||
      !plus2_spatial_rank4_shape(projected_sum_value, modes, aggregate_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(projected_sum_tangent, modes, aggregate_count,
                                 radial_points, theta_points) ||
      !plus2_spatial_rank4_shape(outer_derivative_value, modes,
                                 derivative_count, radial_points,
                                 theta_points) ||
      !plus2_spatial_rank4_shape(outer_derivative_tangent, modes,
                                 derivative_count, radial_points,
                                 theta_points)) {
    throw std::invalid_argument(
        "spin +2 spatial outer input extents or activation are invalid");
  }

  const auto target_indices = workspace.target_indices();
  const auto source_value = workspace.source_value();
  const auto source_tangent = workspace.source_tangent();
  const auto forcing_value = workspace.forcing_value();
  const auto forcing_tangent = workspace.forcing_tangent();
  const std::size_t target_points =
      workspace.target_count() * radial_points * theta_points;
  const detail::Plus2SpatialOuterFunctor functor{
      radial_grid,
      parameters,
      cos_theta.data(),
      sin_theta.data(),
      projected_sum_value.data(),
      projected_sum_tangent.data(),
      outer_derivative_value.data(),
      outer_derivative_tangent.data(),
      target_indices.data(),
      source_value.data(),
      source_tangent.data(),
      forcing_value.data(),
      forcing_tangent.data(),
      source_activation_multiplier,
      radial_points,
      theta_points};
  Kokkos::parallel_for("teuk_plus2_spatial_outer_source",
                       Kokkos::RangePolicy<ExecutionSpace>(execution, 0,
                                                           target_points),
                       functor);
}

}  // namespace teuk
