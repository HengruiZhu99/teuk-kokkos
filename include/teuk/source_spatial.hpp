#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/ghp.hpp"
#include "teuk/modes.hpp"
#include "teuk/second_order.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class SpatialSourceField : std::size_t {
  F = 0,
  G = 1,
  H = 2,
  Lambda = 3,
  Pi = 4,
  B = 5,
  C = 6,
  U = 7,
  Count = 8,
};

enum class SpatialSourceDerivative : std::size_t {
  Delta1F = 0,
  Delta3U = 1,
  Eth2C = 2,
  EthPrime2CSharp = 3,
  Eth1B = 4,
  Delta2C = 5,
  Delta2G = 6,
  Eth2G = 7,
  EthPrime1F = 8,
  Delta2CSharp = 9,
  EthPrime1BSharp = 10,
  Count = 11,
};

enum class SpatialInnerSourceComponent : std::size_t { D = 0, T = 1, Count = 2 };

struct SpatialSourcePair {
  std::size_t index1 = 0;
  std::size_t index2 = 0;
  std::size_t target_index = 0;
};

using SpatialSourceFieldView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using SpatialSourceDerivativeView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using SpatialInnerSourceView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using SpatialPairSourceView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using SpatialSourcePairView = Kokkos::View<SpatialSourcePair*, MemorySpace>;
using SpatialSourceIndexView = Kokkos::View<std::size_t*, MemorySpace>;
using SpatialThetaView = Kokkos::View<Real*, MemorySpace>;
using SpatialOuterSourceView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;

// Device-resident deterministic ordered-pair workspace. The input fields and
// their derivatives are caller-owned stage views; outputs retain both summed
// target sources and every ordered-pair contribution for diagnostics.
class SpatialInnerSourceWorkspace {
 public:
  SpatialInnerSourceWorkspace(const ModeRegistry& registry,
                              const std::size_t radial_points,
                              const std::size_t theta_points,
                              const std::string& label = "spatial_source")
      : mode_count_(registry.size()),
        radial_points_(radial_points),
        theta_points_(theta_points),
        pairs_(label + "_pairs", registry.ordered_pairs().size()),
        sharp_indices_(label + "_sharp_indices", registry.size()),
        target_indices_(label + "_target_indices", registry.targets().size()),
        pair_offsets_(label + "_pair_offsets", registry.targets().size() + 1),
        summed_(label + "_summed", registry.size(),
                static_cast<std::size_t>(SpatialInnerSourceComponent::Count),
                radial_points, theta_points),
        per_pair_(label + "_per_pair", registry.ordered_pairs().size(),
                  static_cast<std::size_t>(
                      SpatialInnerSourceComponent::Count),
                  radial_points, theta_points) {
    if (!registry.is_closed_under_sharp()) {
      throw std::invalid_argument(
          "spatial source modes must be closed under m -> -m");
    }
    if (radial_points == 0 || theta_points == 0) {
      throw std::invalid_argument("spatial source extents must be nonzero");
    }
    auto host_pairs = Kokkos::create_mirror_view(pairs_);
    auto host_sharp = Kokkos::create_mirror_view(sharp_indices_);
    auto host_targets = Kokkos::create_mirror_view(target_indices_);
    auto host_offsets = Kokkos::create_mirror_view(pair_offsets_);
    for (std::size_t i = 0; i < registry.ordered_pairs().size(); ++i) {
      const ModePair& pair = registry.ordered_pairs()[i];
      host_pairs(i) = {pair.index1, pair.index2, pair.target_index};
    }
    for (std::size_t i = 0; i < registry.size(); ++i) {
      host_sharp(i) = registry.sharp_index(registry.modes()[i]);
    }
    for (std::size_t i = 0; i < registry.targets().size(); ++i) {
      host_targets(i) = registry.index(registry.targets()[i]);
      const auto [begin, end] = registry.pair_range(registry.targets()[i]);
      host_offsets(i) = begin;
      host_offsets(i + 1) = end;
    }
    Kokkos::deep_copy(pairs_, host_pairs);
    Kokkos::deep_copy(sharp_indices_, host_sharp);
    Kokkos::deep_copy(target_indices_, host_targets);
    Kokkos::deep_copy(pair_offsets_, host_offsets);
  }

  [[nodiscard]] std::size_t mode_count() const { return mode_count_; }
  [[nodiscard]] std::size_t radial_points() const { return radial_points_; }
  [[nodiscard]] std::size_t theta_points() const { return theta_points_; }
  [[nodiscard]] std::size_t target_count() const {
    return target_indices_.extent(0);
  }
  [[nodiscard]] std::size_t pair_count() const { return pairs_.extent(0); }
  [[nodiscard]] SpatialSourcePairView pairs() const { return pairs_; }
  [[nodiscard]] SpatialSourceIndexView sharp_indices() const {
    return sharp_indices_;
  }
  [[nodiscard]] SpatialSourceIndexView target_indices() const {
    return target_indices_;
  }
  [[nodiscard]] SpatialSourceIndexView pair_offsets() const {
    return pair_offsets_;
  }
  [[nodiscard]] SpatialInnerSourceView summed() const { return summed_; }
  [[nodiscard]] SpatialPairSourceView per_pair() const { return per_pair_; }

 private:
  std::size_t mode_count_;
  std::size_t radial_points_;
  std::size_t theta_points_;
  SpatialSourcePairView pairs_;
  SpatialSourceIndexView sharp_indices_;
  SpatialSourceIndexView target_indices_;
  SpatialSourceIndexView pair_offsets_;
  SpatialInnerSourceView summed_;
  SpatialPairSourceView per_pair_;
};

template <class ExecutionSpace>
void evaluate_spatial_inner_source(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const SpatialThetaView& cos_theta,
    const SpatialThetaView& sin_theta, const SpatialSourceFieldView& fields,
    const SpatialSourceDerivativeView& derivatives,
    SpatialInnerSourceWorkspace& workspace) {
  const std::size_t modes = workspace.mode_count();
  const std::size_t radial_points = workspace.radial_points();
  const std::size_t theta_points = workspace.theta_points();
  if (radial_grid.size() != radial_points || cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points || fields.extent(0) != modes ||
      fields.extent(1) !=
          static_cast<std::size_t>(SpatialSourceField::Count) ||
      fields.extent(2) != radial_points || fields.extent(3) != theta_points ||
      derivatives.extent(0) != modes ||
      derivatives.extent(1) !=
          static_cast<std::size_t>(SpatialSourceDerivative::Count) ||
      derivatives.extent(2) != radial_points ||
      derivatives.extent(3) != theta_points) {
    throw std::invalid_argument("spatial source view extents do not match");
  }

  const auto pairs = workspace.pairs();
  const auto sharp = workspace.sharp_indices();
  const auto targets = workspace.target_indices();
  const auto offsets = workspace.pair_offsets();
  const auto summed = workspace.summed();
  const auto per_pair = workspace.per_pair();
  // Non-target slots are intentionally zero rather than stale. Target nodal
  // values must still be analyzed/projected to the configured angular band by
  // the caller before an outer angular derivative is applied.
  Kokkos::deep_copy(execution, summed, Complex(0.0, 0.0));
  const std::size_t target_points =
      workspace.target_count() * radial_points * theta_points;
  Kokkos::parallel_for(
      "teuk_spatial_inner_source",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, target_points),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t target_plane = radial_points * theta_points;
        const std::size_t target_slot = flat / target_plane;
        const std::size_t within_target = flat - target_slot * target_plane;
        const std::size_t radial = within_target / theta_points;
        const std::size_t theta = within_target - radial * theta_points;
        const std::size_t target = targets(target_slot);
        Complex total_D(0.0, 0.0);
        Complex total_T(0.0, 0.0);
        const double radius = radial_grid.coordinate(radial);
        const KerrBackgroundPoint background = kerr_background_point(
            parameters, radius, cos_theta(theta), sin_theta(theta));

        for (std::size_t pair_index = offsets(target_slot);
             pair_index < offsets(target_slot + 1); ++pair_index) {
          const SpatialSourcePair pair = pairs(pair_index);
          const std::size_t m1 = pair.index1;
          const std::size_t m2 = pair.index2;
          const std::size_t sharp1 = sharp(m1);
          const std::size_t sharp2 = sharp(m2);
          const auto field = [&](const std::size_t mode,
                                 const SpatialSourceField component) {
            return fields(mode, static_cast<std::size_t>(component), radial,
                          theta);
          };
          const auto derivative = [&](const std::size_t mode,
                                      const SpatialSourceDerivative component) {
            return derivatives(mode, static_cast<std::size_t>(component),
                               radial, theta);
          };
          const OrderedPairFields point_fields{
              field(m1, SpatialSourceField::F),
              field(m1, SpatialSourceField::G),
              field(m1, SpatialSourceField::Lambda),
              field(m1, SpatialSourceField::Pi),
              field(m1, SpatialSourceField::B),
              field(m1, SpatialSourceField::C),
              field(m1, SpatialSourceField::U),
              field(m2, SpatialSourceField::F),
              field(m2, SpatialSourceField::G),
              field(m2, SpatialSourceField::H),
              field(m2, SpatialSourceField::B),
              field(m2, SpatialSourceField::C),
              field(m2, SpatialSourceField::U),
              Kokkos::conj(field(sharp2, SpatialSourceField::U)),
              Kokkos::conj(field(sharp2, SpatialSourceField::C)),
              Kokkos::conj(field(sharp1, SpatialSourceField::C)),
              Kokkos::conj(field(sharp1, SpatialSourceField::B)),
              Kokkos::conj(field(sharp2, SpatialSourceField::Pi)),
              Kokkos::conj(field(sharp2, SpatialSourceField::B))};
          const OrderedPairDerivatives point_derivatives{
              derivative(m2, SpatialSourceDerivative::Delta1F),
              derivative(m2, SpatialSourceDerivative::Delta3U),
              derivative(m2, SpatialSourceDerivative::Eth2C),
              derivative(m2, SpatialSourceDerivative::EthPrime2CSharp),
              derivative(m2, SpatialSourceDerivative::Eth1B),
              derivative(m2, SpatialSourceDerivative::Delta2C),
              derivative(m2, SpatialSourceDerivative::Delta2G),
              derivative(m2, SpatialSourceDerivative::Eth2G),
              derivative(m2, SpatialSourceDerivative::EthPrime1F),
              derivative(m2, SpatialSourceDerivative::Delta2CSharp),
              derivative(m2, SpatialSourceDerivative::EthPrime1BSharp)};
          const InnerSource source = corrected_ordered_pair_source(
              radius, background, point_fields, point_derivatives);
          per_pair(pair_index,
                   static_cast<std::size_t>(SpatialInnerSourceComponent::D),
                   radial, theta) = source.D;
          per_pair(pair_index,
                   static_cast<std::size_t>(SpatialInnerSourceComponent::T),
                   radial, theta) = source.T;
          total_D += source.D;
          total_T += source.T;
        }
        summed(target,
               static_cast<std::size_t>(SpatialInnerSourceComponent::D),
               radial, theta) = total_D;
        summed(target,
               static_cast<std::size_t>(SpatialInnerSourceComponent::T),
               radial, theta) = total_T;
      });
}

// Complete outer operator from projected target D/T values and their analytic
// common-stage tangents. lowered_T is L_{-1}T in nodal representation and must
// be computed after angular projection/filtering of T. Both output views have
// ordering (stored target mode,radial,theta); non-target slots remain zero when
// their inputs are zero.
template <class ExecutionSpace>
void evaluate_spatial_outer_source(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const SpatialThetaView& cos_theta,
    const SpatialThetaView& sin_theta,
    const SpatialInnerSourceView& inner_source,
    const SpatialInnerSourceView& inner_source_dt,
    const SpatialOuterSourceView& lowered_T,
    const SpatialOuterSourceView& source_over_r3,
    const SpatialOuterSourceView& forcing,
    const RadialDiscretization discretization = RadialDiscretization::D42) {
  const std::size_t mode_count = inner_source.extent(0);
  const std::size_t radial_points = radial_grid.size();
  const std::size_t theta_points = cos_theta.extent(0);
  const auto valid_inner = [&](const SpatialInnerSourceView& view) {
    return view.extent(0) == mode_count &&
           view.extent(1) == static_cast<std::size_t>(
                                  SpatialInnerSourceComponent::Count) &&
           view.extent(2) == radial_points &&
           view.extent(3) == theta_points;
  };
  const auto valid_outer = [&](const SpatialOuterSourceView& view) {
    return view.extent(0) == mode_count && view.extent(1) == radial_points &&
           view.extent(2) == theta_points;
  };
  if (radial_points < radial_minimum_points(discretization) ||
      sin_theta.extent(0) != theta_points || !valid_inner(inner_source) ||
      !valid_inner(inner_source_dt) || !valid_outer(lowered_T) ||
      !valid_outer(source_over_r3) || !valid_outer(forcing)) {
    throw std::invalid_argument("spatial outer source view extents do not match");
  }

  constexpr std::size_t D =
      static_cast<std::size_t>(SpatialInnerSourceComponent::D);
  constexpr std::size_t T =
      static_cast<std::size_t>(SpatialInnerSourceComponent::T);
  const double inverse_spacing = 1.0 / radial_grid.spacing();
  const std::size_t radial_stride = inner_source.stride(2);
  const std::size_t total = mode_count * radial_points * theta_points;
  Kokkos::parallel_for(
      "teuk_spatial_outer_source",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_plane = radial_points * theta_points;
        const std::size_t mode = flat / mode_plane;
        const std::size_t within_mode = flat - mode * mode_plane;
        const std::size_t radial = within_mode / theta_points;
        const std::size_t theta = within_mode - radial * theta_points;
        const double radius = radial_grid.coordinate(radial);
        const Complex D_value = inner_source(mode, D, radial, theta);
        const Complex T_value = inner_source(mode, T, radial, theta);
        const Complex radial_D = radial_first_derivative_strided_at(
            discretization,
            &inner_source(mode, D, 0, theta), radial_points, radial,
            inverse_spacing, radial_stride);
        const Complex delta3_D = delta_n_point(
            D_value, inner_source_dt(mode, D, radial, theta), radial_D, 3,
            radius, parameters.mass, parameters.compactification_length);
        const Complex ethprime3_T = ethprime_n_point(
            T_value, inner_source_dt(mode, T, radial, theta),
            lowered_T(mode, radial, theta), -1, -2, radius,
            sin_theta(theta), cos_theta(theta), parameters.spin,
            parameters.compactification_length);
        const KerrBackgroundPoint background = kerr_background_point(
            parameters, radius, cos_theta(theta), sin_theta(theta));
        const Complex source = outer_source_over_r3(
            radius, background, InnerSource{D_value, T_value},
            OuterSourceDerivatives{delta3_D, ethprime3_T});
        source_over_r3(mode, radial, theta) = source;
        forcing(mode, radial, theta) = coordinate_second_order_forcing(
            radius, cos_theta(theta), parameters.spin,
            parameters.compactification_length, source);
      });
}

// Variant for a caller that already applied the complete stage-local GHP
// eth-prime operator to projected T. This is the natural composition point for
// SignedModeAngularCoordinator::ethprime and avoids exposing its intermediate
// lowered modal field.
template <class ExecutionSpace>
void evaluate_spatial_outer_source_from_ethprime(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const SpatialThetaView& cos_theta,
    const SpatialThetaView& sin_theta,
    const SpatialInnerSourceView& inner_source,
    const SpatialInnerSourceView& inner_source_dt,
    const SpatialOuterSourceView& ethprime3_T,
    const SpatialOuterSourceView& source_over_r3,
    const SpatialOuterSourceView& forcing,
    const RadialDiscretization discretization = RadialDiscretization::D42) {
  const std::size_t mode_count = inner_source.extent(0);
  const std::size_t radial_points = radial_grid.size();
  const std::size_t theta_points = cos_theta.extent(0);
  const auto valid_inner = [&](const SpatialInnerSourceView& view) {
    return view.extent(0) == mode_count &&
           view.extent(1) == static_cast<std::size_t>(
                                  SpatialInnerSourceComponent::Count) &&
           view.extent(2) == radial_points &&
           view.extent(3) == theta_points;
  };
  const auto valid_outer = [&](const SpatialOuterSourceView& view) {
    return view.extent(0) == mode_count && view.extent(1) == radial_points &&
           view.extent(2) == theta_points;
  };
  if (radial_points < radial_minimum_points(discretization) ||
      sin_theta.extent(0) != theta_points || !valid_inner(inner_source) ||
      !valid_inner(inner_source_dt) || !valid_outer(ethprime3_T) ||
      !valid_outer(source_over_r3) || !valid_outer(forcing)) {
    throw std::invalid_argument("spatial outer source view extents do not match");
  }
  constexpr std::size_t D =
      static_cast<std::size_t>(SpatialInnerSourceComponent::D);
  constexpr std::size_t T =
      static_cast<std::size_t>(SpatialInnerSourceComponent::T);
  const double inverse_spacing = 1.0 / radial_grid.spacing();
  const std::size_t radial_stride = inner_source.stride(2);
  const std::size_t total = mode_count * radial_points * theta_points;
  Kokkos::parallel_for(
      "teuk_spatial_outer_source_from_ethprime",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_plane = radial_points * theta_points;
        const std::size_t mode = flat / mode_plane;
        const std::size_t within_mode = flat - mode * mode_plane;
        const std::size_t radial = within_mode / theta_points;
        const std::size_t theta = within_mode - radial * theta_points;
        const double radius = radial_grid.coordinate(radial);
        const Complex D_value = inner_source(mode, D, radial, theta);
        const Complex T_value = inner_source(mode, T, radial, theta);
        const Complex radial_D = radial_first_derivative_strided_at(
            discretization,
            &inner_source(mode, D, 0, theta), radial_points, radial,
            inverse_spacing, radial_stride);
        const Complex delta3_D = delta_n_point(
            D_value, inner_source_dt(mode, D, radial, theta), radial_D, 3,
            radius, parameters.mass, parameters.compactification_length);
        const KerrBackgroundPoint background = kerr_background_point(
            parameters, radius, cos_theta(theta), sin_theta(theta));
        const Complex source = outer_source_over_r3(
            radius, background, InnerSource{D_value, T_value},
            OuterSourceDerivatives{delta3_D,
                                   ethprime3_T(mode, radial, theta)});
        source_over_r3(mode, radial, theta) = source;
        forcing(mode, radial, theta) = coordinate_second_order_forcing(
            radius, cos_theta(theta), parameters.spin,
            parameters.compactification_length, source);
      });
}

}  // namespace teuk
