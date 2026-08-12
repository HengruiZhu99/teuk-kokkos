#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/angular_coordinator.hpp"
#include "teuk/background.hpp"
#include "teuk/ghp.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_source_value_spatial.hpp"
#include "teuk/radial_discretization.hpp"

namespace teuk {

using Plus2OuterConstAggregateView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;

// The pair sum and its analytic J/K tangent belong to one common RK stage.
// The explicit generation prevents a caller from pairing a current write
// target with cached nonlinear sums from a preceding stage.
struct Plus2OuterSpatialStage {
  std::uint64_t generation = 0;
  Plus2OuterConstAggregateView summed_value;
  Plus2OuterConstAggregateView summed_jk_tangent;
};

namespace plus2_outer_spatial_detail {

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

struct SetReadyFunctor {
  std::uint8_t* ready;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t) const {
    ready[0] = static_cast<std::uint8_t>(1);
  }
};

struct ValidateInputsFunctor {
  const Complex* value;
  const Complex* tangent;
  std::uint8_t* ready;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t value_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t tangent_count =
        static_cast<std::size_t>(Plus2ProductionJkAggregate::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = 0; field < value_count; ++field) {
      const Complex x = value[flat4(mode, field, radial, theta, value_count,
                                    radial_count, theta_count)];
      valid = valid && Kokkos::isfinite(x.real()) &&
              Kokkos::isfinite(x.imag());
    }
    for (std::size_t field = 0; field < tangent_count; ++field) {
      const Complex x = tangent[flat4(mode, field, radial, theta,
                                      tangent_count, radial_count,
                                      theta_count)];
      valid = valid && Kokkos::isfinite(x.real()) &&
              Kokkos::isfinite(x.imag());
    }
    if (!valid) {
      Kokkos::atomic_exchange(ready, static_cast<std::uint8_t>(0));
    }
  }
};

struct Thorn5Functor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* cos_theta;
  const Real* sin_theta;
  const int* modes;
  const Complex* projected;
  const Complex* projected_tangent;
  Complex* outer;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t tangent_count =
        static_cast<std::size_t>(Plus2ProductionJkAggregate::Count);
    constexpr std::size_t outer_count =
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
    constexpr std::size_t j =
        static_cast<std::size_t>(Plus2SpatialAggregate::J);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t value_index = flat4(
        mode, j, radial, theta, aggregate_count, radial_count, theta_count);
    const Complex value = projected[value_index];
    const Complex tangent = projected_tangent[flat4(
        mode, 0, radial, theta, tangent_count, radial_count, theta_count)];
    const Complex radial_derivative = radial_first_derivative_strided_at(
        RadialDiscretization::D105,
        &projected[flat4(mode, j, 0, theta, aggregate_count, radial_count,
                         theta_count)],
        radial_count, radial, inverse_spacing, theta_count);
    const double radius = grid.coordinate(radial);
    const KerrBackgroundPoint background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    outer[flat4(mode,
                static_cast<std::size_t>(
                    Plus2SpatialOuterDerivative::Thorn5J),
                radial, theta, outer_count, radial_count, theta_count)] =
        thorn_n_point(value, tangent, radial_derivative, 5, 2, 1,
                      modes[mode], radius, cos_theta[theta], parameters.mass,
                      parameters.spin, parameters.compactification_length,
                      background.epsilon0);
    outer[flat4(
        mode,
        static_cast<std::size_t>(
            Plus2SpatialOuterDerivative::
                RegularizedThorn5JMinusOpticalJOverR),
        radial, theta, outer_count, radial_count, theta_count)] =
        plus2_regularized_thorn5_j_minus_optical_over_r(
            value, tangent, radial_derivative, modes[mode], radius,
            cos_theta[theta], parameters, background);
  }
};

struct FinalizeFunctor {
  Complex* projected;
  Complex* outer;
  std::uint64_t* projected_stamps;
  std::uint64_t* outer_stamps;
  const std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t outer_count =
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = ready[0] != 0;
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      aggregate_count, radial_count,
                                      theta_count);
      const Complex x = projected[index];
      valid = valid && Kokkos::isfinite(x.real()) &&
              Kokkos::isfinite(x.imag());
    }
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, outer_count,
                                      radial_count, theta_count);
      const Complex x = outer[index];
      valid = valid && Kokkos::isfinite(x.real()) &&
              Kokkos::isfinite(x.imag());
    }
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      aggregate_count, radial_count,
                                      theta_count);
      if (!valid) projected[index] = Complex{};
      projected_stamps[index] = valid ? generation : 0;
    }
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, outer_count,
                                      radial_count, theta_count);
      if (!valid) outer[index] = Complex{};
      outer_stamps[index] = valid ? generation : 0;
    }
  }
};

static_assert(std::is_trivially_copyable_v<SetReadyFunctor>);
static_assert(std::is_trivially_copyable_v<ValidateInputsFunctor>);
static_assert(std::is_trivially_copyable_v<Thorn5Functor>);
static_assert(std::is_trivially_copyable_v<FinalizeFunctor>);
static_assert(sizeof(Thorn5Functor) < 1800);

template <class LeftView, class RightView>
bool same_allocation(const LeftView& left, const RightView& right) {
  return static_cast<const void*>(left.data()) ==
         static_cast<const void*>(right.data());
}

}  // namespace plus2_outer_spatial_detail

// Concrete allocation-owning outer graph. Construction performs every host
// lookup/allocation/copy. evaluate() only enqueues kernels and uses the exact
// signed-m angular coordinator for retained-band projection and eth_6.
template <class ExecSpace = ExecutionSpace>
class Plus2SourceOuterSpatialProducer {
 public:
  using execution_space = ExecSpace;
  using scratch_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
  using real_view = Kokkos::View<Real*, MemorySpace>;
  using mode_view = Kokkos::View<int*, MemorySpace>;

  Plus2SourceOuterSpatialProducer(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& grid, const KerrParameters& parameters,
      const int ell_max, const Plus2SpatialThetaView& cos_theta,
      const Plus2SpatialThetaView& sin_theta,
      const std::string& label = "plus2_source_outer_spatial",
      const RadialDiscretization discretization =
          RadialDiscretization::D105)
      : registry_(registry),
        grid_(grid),
        parameters_(parameters),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        discretization_(discretization),
        ell_max_(ell_max),
        radius_(label + "_radius", grid.size()),
        modes_(label + "_modes", registry.size()),
        projected_tangent_(
            label + "_projected_tangent", registry.size(),
            static_cast<std::size_t>(Plus2ProductionJkAggregate::Count),
            grid.size(), sin_theta.extent(0)),
        ready_(label + "_ready", 1),
        j_angular_(execution, registry, 2, 1, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters, registry.targets()),
        k_angular_(execution, registry, 1, 2, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters, registry.targets()),
        q_angular_(execution, registry, 2, 2, ell_max,
                   static_cast<int>(sin_theta.extent(0)), grid.size(),
                   parameters, registry.targets()) {
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
          "spin +2 outer producer requires a closed D10-5 geometry");
    }
    auto host_radius = Kokkos::create_mirror_view(radius_);
    auto host_modes = Kokkos::create_mirror_view(modes_);
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      host_radius(radial) = grid.coordinate(radial);
    }
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_modes(mode) = registry.modes()[mode];
    }
    Kokkos::deep_copy(execution, radius_, host_radius);
    Kokkos::deep_copy(execution, modes_, host_modes);
  }

  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return discretization_;
  }
  [[nodiscard]] bool matches_configuration(
      const ModeRegistry& registry, const UniformRadialGrid& grid,
      const KerrParameters& parameters, const int ell_max,
      const Plus2SpatialThetaView& cos_theta,
      const Plus2SpatialThetaView& sin_theta) const noexcept {
    return discretization_ == RadialDiscretization::D105 &&
           ell_max_ == ell_max && registry_.modes() == registry.modes() &&
           registry_.parents() == registry.parents() &&
           registry_.targets() == registry.targets() &&
           grid_.size() == grid.size() &&
           grid_.lower_radius() == grid.lower_radius() &&
           grid_.upper_radius() == grid.upper_radius() &&
           parameters_.mass == parameters.mass &&
           parameters_.spin == parameters.spin &&
           parameters_.compactification_length ==
               parameters.compactification_length &&
           cos_theta_.data() == cos_theta.data() &&
           sin_theta_.data() == sin_theta.data();
  }
  [[nodiscard]] scratch_view projected_tangent() const {
    return projected_tangent_;
  }
  [[nodiscard]] Kokkos::View<const std::uint8_t*, MemorySpace> readiness()
      const {
    return ready_;
  }

  template <class WriteTarget>
  void evaluate(const execution_space& execution,
                const Plus2OuterSpatialStage& stage,
                const WriteTarget& target) {
    validate(stage, target);
    using namespace plus2_outer_spatial_detail;
    const std::size_t radial = grid_.size();
    const std::size_t theta = sin_theta_.extent(0);
    const std::size_t points = registry_.size() * radial * theta;
    Kokkos::parallel_for(
        "plus2_outer_set_ready",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        SetReadyFunctor{ready_.data()});
    Kokkos::parallel_for(
        "plus2_outer_validate_inputs",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        ValidateInputsFunctor{stage.summed_value.data(),
                              stage.summed_jk_tangent.data(), ready_.data(),
                              radial, theta});

    constexpr std::size_t J =
        static_cast<std::size_t>(Plus2SpatialAggregate::J);
    constexpr std::size_t K =
        static_cast<std::size_t>(Plus2SpatialAggregate::K);
    constexpr std::size_t Q =
        static_cast<std::size_t>(Plus2SpatialAggregate::Q);
    j_angular_.project(execution, stage.summed_value, J,
                       target.projected_sum_value, J);
    k_angular_.project(execution, stage.summed_value, K,
                       target.projected_sum_value, K);
    q_angular_.project(execution, stage.summed_value, Q,
                       target.projected_sum_value, Q);
    j_angular_.project(execution, stage.summed_jk_tangent, 0,
                       projected_tangent_, 0);
    k_angular_.project(execution, stage.summed_jk_tangent, 1,
                       projected_tangent_, 1);

    Kokkos::parallel_for(
        "plus2_outer_thorn5_j",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        Thorn5Functor{grid_, parameters_, cos_theta_.data(),
                      sin_theta_.data(), modes_.data(),
                      target.projected_sum_value.data(),
                      projected_tangent_.data(),
                      target.outer_derivative_value.data(), radial, theta,
                      1.0 / grid_.spacing()});
    k_angular_.eth(
        execution, target.projected_sum_value, K, projected_tangent_, 1,
        radius_, sin_theta_, cos_theta_, target.outer_derivative_value,
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Eth6K));
    Kokkos::parallel_for(
        "plus2_outer_finalize",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        FinalizeFunctor{target.projected_sum_value.data(),
                        target.outer_derivative_value.data(),
                        target.projected_sum_value_stamps.data(),
                        target.outer_derivative_value_stamps.data(),
                        ready_.data(), target.generation, radial, theta});
  }

 private:
  template <class WriteTarget>
  void validate(const Plus2OuterSpatialStage& stage,
                const WriteTarget& target) const {
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t tangent_count =
        static_cast<std::size_t>(Plus2ProductionJkAggregate::Count);
    constexpr std::size_t outer_count =
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
    const std::size_t modes = registry_.size();
    const std::size_t radial = grid_.size();
    const std::size_t theta = sin_theta_.extent(0);
    const auto shape = [&](const auto& view, const std::size_t fields) {
      return view.extent(0) == modes && view.extent(1) == fields &&
             view.extent(2) == radial && view.extent(3) == theta;
    };
    using plus2_outer_spatial_detail::same_allocation;
    if (stage.generation == 0 || stage.generation != target.generation ||
        !shape(stage.summed_value, aggregate_count) ||
        !shape(stage.summed_jk_tangent, tangent_count) ||
        !shape(target.projected_sum_value, aggregate_count) ||
        !shape(target.outer_derivative_value, outer_count) ||
        !shape(target.projected_sum_value_stamps, aggregate_count) ||
        !shape(target.outer_derivative_value_stamps, outer_count) ||
        same_allocation(stage.summed_value, stage.summed_jk_tangent) ||
        same_allocation(stage.summed_value, target.projected_sum_value) ||
        same_allocation(stage.summed_value, target.outer_derivative_value) ||
        same_allocation(stage.summed_jk_tangent,
                        target.projected_sum_value) ||
        same_allocation(stage.summed_jk_tangent,
                        target.outer_derivative_value) ||
        same_allocation(target.projected_sum_value,
                        target.outer_derivative_value) ||
        same_allocation(target.projected_sum_value_stamps,
                        target.outer_derivative_value_stamps) ||
        same_allocation(target.projected_sum_value, projected_tangent_) ||
        same_allocation(target.outer_derivative_value,
                        projected_tangent_)) {
      throw std::invalid_argument(
          "spin +2 outer producer stage extent, generation, or alias is invalid");
    }
  }

  ModeRegistry registry_;
  UniformRadialGrid grid_;
  KerrParameters parameters_;
  Plus2SpatialThetaView cos_theta_;
  Plus2SpatialThetaView sin_theta_;
  RadialDiscretization discretization_;
  int ell_max_;
  real_view radius_;
  mode_view modes_;
  scratch_view projected_tangent_;
  Kokkos::View<std::uint8_t*, MemorySpace> ready_;
  SignedModeAngularCoordinator<execution_space> j_angular_;
  SignedModeAngularCoordinator<execution_space> k_angular_;
  SignedModeAngularCoordinator<execution_space> q_angular_;
};

}  // namespace teuk
