#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/angular_coordinate_device.hpp"
#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_linear_psi0.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

namespace plus2_linear_spatial_detail {

KOKKOS_INLINE_FUNCTION
std::size_t flat_metric(const std::size_t mode, const std::size_t component,
                        const std::size_t radial, const std::size_t theta,
                        const std::size_t radial_count,
                        const std::size_t theta_count) {
  constexpr std::size_t components = 3;
  return ((mode * components + component) * radial_count + radial) *
             theta_count +
         theta;
}

KOKKOS_INLINE_FUNCTION
std::size_t flat_field(const std::size_t mode, const std::size_t radial,
                       const std::size_t theta,
                       const std::size_t radial_count,
                       const std::size_t theta_count) {
  return (mode * radial_count + radial) * theta_count + theta;
}

template <class InputScalar>
struct PackMetricFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* sin_theta;
  const Real* cos_theta;
  const InputScalar* reconstruction;
  const InputScalar* tangent;
  const InputScalar* second_tangent;
  const std::size_t* sharp;
  Complex* stage_metric;
  Complex* tangent_metric;
  Complex* second_metric;
  std::size_t input_fields;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t offset_B;
  std::size_t offset_C;
  std::size_t offset_U;

  KOKKOS_INLINE_FUNCTION
  std::size_t input_index(const std::size_t mode, const std::size_t field,
                          const std::size_t radial,
                          const std::size_t theta) const {
    return ((mode * input_fields + field) * radial_count + radial) *
               theta_count +
           theta;
  }

  KOKKOS_INLINE_FUNCTION
  Plus2OrgMetricFields pack(const InputScalar* input,
                            const std::size_t mode,
                            const std::size_t sharp_mode,
                            const std::size_t radial,
                            const std::size_t theta, const double radius,
                            const Complex& mu0) const {
    return plus2_org_metric_from_reconstruction(
        radius, mu0,
        Kokkos::conj(input[input_index(sharp_mode, offset_B, radial, theta)]),
        Kokkos::conj(input[input_index(sharp_mode, offset_C, radial, theta)]),
        input[input_index(mode, offset_U, radial, theta)]);
  }

  KOKKOS_INLINE_FUNCTION
  void store(Complex* output, const std::size_t mode,
             const std::size_t radial, const std::size_t theta,
             const Plus2OrgMetricFields& fields) const {
    output[flat_metric(mode, 0, radial, theta, radial_count, theta_count)] =
        fields.h_ll;
    output[flat_metric(mode, 1, radial, theta, radial_count, theta_count)] =
        fields.h_lm;
    output[flat_metric(mode, 2, radial, theta, radial_count, theta_count)] =
        fields.h_mm;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t sharp_mode = sharp[mode];
    const double radius = grid.coordinate(radial);
    const auto background = kerr_background_point(
        parameters, radius, cos_theta[theta], sin_theta[theta]);
    store(stage_metric, mode, radial, theta,
          pack(reconstruction, mode, sharp_mode, radial, theta, radius,
               background.mu0));
    store(tangent_metric, mode, radial, theta,
          pack(tangent, mode, sharp_mode, radial, theta, radius,
               background.mu0));
    store(second_metric, mode, radial, theta,
          pack(second_tangent, mode, sharp_mode, radial, theta, radius,
               background.mu0));
  }
};

struct FirstRadialFunctor {
  const Complex* stage;
  const Complex* tangent;
  Complex* radial;
  Complex* time_radial;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;
  RadialDiscretization discretization;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    const std::size_t line_plane = radial_count * theta_count;
    const std::size_t mode_component = flat / line_plane;
    const std::size_t within = flat - mode_component * line_plane;
    const std::size_t mode = mode_component / 3;
    const std::size_t component = mode_component - mode * 3;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta = within - radial_index * theta_count;
    const std::size_t base =
        flat_metric(mode, component, 0, theta, radial_count, theta_count);
    const std::size_t output = flat_metric(mode, component, radial_index,
                                           theta, radial_count, theta_count);
    radial[output] = radial_first_derivative_strided_at(
        discretization, stage + base, radial_count, radial_index,
        inverse_spacing, theta_count);
    time_radial[output] = radial_first_derivative_strided_at(
        discretization, tangent + base, radial_count, radial_index,
        inverse_spacing, theta_count);
  }
};

struct SecondRadialFunctor {
  const Complex* radial;
  Complex* radial_radial;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;
  RadialDiscretization discretization;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    const std::size_t line_plane = radial_count * theta_count;
    const std::size_t mode_component = flat / line_plane;
    const std::size_t within = flat - mode_component * line_plane;
    const std::size_t mode = mode_component / 3;
    const std::size_t component = mode_component - mode * 3;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta = within - radial_index * theta_count;
    const std::size_t base =
        flat_metric(mode, component, 0, theta, radial_count, theta_count);
    const std::size_t output = flat_metric(mode, component, radial_index,
                                           theta, radial_count, theta_count);
    radial_radial[output] = radial_first_derivative_strided_at(
        discretization, radial + base, radial_count, radial_index,
        inverse_spacing, theta_count);
  }
};

struct EvaluatePointFunctor {
  UniformRadialGrid grid;
  KerrParameters parameters;
  const Real* sin_theta;
  const Real* cos_theta;
  const int* modes;
  const Complex* stage;
  const Complex* tangent;
  const Complex* second_tangent;
  const Complex* radial;
  const Complex* time_radial;
  const Complex* radial_radial;
  const Complex* theta;
  const Complex* time_theta;
  const Complex* radial_theta;
  const Complex* theta_theta;
  Complex* raw;
  Complex* z_plus;
  std::uint8_t* valid;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION
  Plus2OrgMetricFields load(const Complex* input, const std::size_t mode,
                            const std::size_t radial_index,
                            const std::size_t theta_index) const {
    return {input[flat_metric(mode, 0, radial_index, theta_index,
                              radial_count, theta_count)],
            input[flat_metric(mode, 1, radial_index, theta_index,
                              radial_count, theta_count)],
            input[flat_metric(mode, 2, radial_index, theta_index,
                              radial_count, theta_count)]};
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta_index = within - radial_index * theta_count;
    Plus2OrgMetricStage point_stage{load(stage, mode, radial_index, theta_index),
                                     load(tangent, mode, radial_index,
                                          theta_index),
                                     load(second_tangent, mode, radial_index,
                                          theta_index)};
    Plus2OrgMetricDerivativeSlots derivatives{};
    derivatives.h_R = load(radial, mode, radial_index, theta_index);
    derivatives.h_TR = load(time_radial, mode, radial_index, theta_index);
    derivatives.h_RR = load(radial_radial, mode, radial_index, theta_index);
    derivatives.h_theta = load(theta, mode, radial_index, theta_index);
    derivatives.h_Ttheta = load(time_theta, mode, radial_index, theta_index);
    derivatives.h_Rtheta = load(radial_theta, mode, radial_index, theta_index);
    derivatives.h_thetatheta =
        load(theta_theta, mode, radial_index, theta_index);
    plus2_fill_modal_azimuthal_derivatives(modes[mode], point_stage,
                                           derivatives);
    const double radius_value = grid.coordinate(radial_index);
    const auto result = evaluate_plus2_linear_psi0(
        parameters, radius_value, sin_theta[theta_index],
        cos_theta[theta_index], point_stage, derivatives);
    const std::size_t output = flat_field(mode, radial_index, theta_index,
                                          radial_count, theta_count);
    raw[output] = result.psi0_code_tetrad;
    const auto regularized = regularize_plus2_linear_psi0(
        result.psi0_code_tetrad, radius_value, cos_theta[theta_index],
        parameters.spin, parameters.compactification_length);
    valid[output] = static_cast<std::uint8_t>(regularized.valid);
    z_plus[output] = regularized.valid ? regularized.z_plus : Complex{};
  }
};

static_assert(std::is_trivially_copyable_v<PackMetricFunctor<Complex>>);
static_assert(std::is_trivially_copyable_v<FirstRadialFunctor>);
static_assert(std::is_trivially_copyable_v<SecondRadialFunctor>);
static_assert(std::is_trivially_copyable_v<EvaluatePointFunctor>);
static_assert(sizeof(PackMetricFunctor<Complex>) < 1800);
static_assert(sizeof(FirstRadialFunctor) < 1800);
static_assert(sizeof(SecondRadialFunctor) < 1800);
static_assert(sizeof(EvaluatePointFunctor) < 1800);

}  // namespace plus2_linear_spatial_detail

enum class Plus2MetricComponent : std::size_t {
  LL = 0,
  LM = 1,
  MM = 2,
  Count = 3,
};

struct Plus2ReconstructionMetricOffsets {
  std::size_t B = 0;
  std::size_t C = 1;
  std::size_t U = 2;
};

template <class ExecSpace = ExecutionSpace>
class Plus2LinearPsi0SpatialWorkspace {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename execution_space::memory_space;
  using metric_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, memory_space>;
  using field_view =
      Kokkos::View<Complex***, Kokkos::LayoutRight, memory_space>;
  using validity_view =
      Kokkos::View<std::uint8_t***, Kokkos::LayoutRight, memory_space>;
  using mode_view = Kokkos::View<int*, memory_space>;
  using index_view = Kokkos::View<std::size_t*, memory_space>;
  using coordinate_plan = DeviceSpinCoordinateDerivativePlan<execution_space>;

  Plus2LinearPsi0SpatialWorkspace(
      const execution_space& execution, const ModeRegistry& registry,
      const std::size_t radial_count, const int ell_max,
      const int theta_count,
      const std::string& label = "plus2_linear_spatial",
      const RadialDiscretization discretization =
          RadialDiscretization::D42)
      : mode_count_(registry.size()),
        radial_count_(radial_count),
        theta_count_(static_cast<std::size_t>(theta_count)),
        discretization_(discretization),
        modes_(label + "_modes", registry.size()),
        sharp_(label + "_sharp", registry.size()),
        stage_(label + "_stage", registry.size(), component_count(),
               radial_count, theta_count_),
        tangent_(label + "_tangent", registry.size(), component_count(),
                 radial_count, theta_count_),
        second_tangent_(label + "_second_tangent", registry.size(),
                        component_count(), radial_count, theta_count_),
        radial_(label + "_radial", registry.size(), component_count(),
                radial_count, theta_count_),
        time_radial_(label + "_time_radial", registry.size(),
                     component_count(), radial_count, theta_count_),
        radial_radial_(label + "_radial_radial", registry.size(),
                       component_count(), radial_count, theta_count_),
        theta_(label + "_theta", registry.size(), component_count(),
               radial_count, theta_count_),
        time_theta_(label + "_time_theta", registry.size(),
                    component_count(), radial_count, theta_count_),
        radial_theta_(label + "_radial_theta", registry.size(),
                      component_count(), radial_count, theta_count_),
        theta_theta_(label + "_theta_theta", registry.size(),
                     component_count(), radial_count, theta_count_),
        raw_psi0_(label + "_raw_psi0", registry.size(), radial_count,
                  theta_count_),
        z_plus_(label + "_z_plus", registry.size(), radial_count,
                theta_count_),
        z_plus_valid_(label + "_z_plus_valid", registry.size(), radial_count,
                      theta_count_) {
    if (!registry.is_closed_under_sharp() ||
        radial_count < radial_minimum_points(discretization) ||
        theta_count <= 0 || ell_max < 2) {
      throw std::invalid_argument(
          "plus2 linear spatial workspace geometry or registry is invalid");
    }
    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_sharp = Kokkos::create_mirror_view(sharp_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_modes(mode) = registry.modes()[mode];
      host_sharp(mode) = registry.sharp_index(registry.modes()[mode]);
    }
    Kokkos::deep_copy(execution, modes_, host_modes);
    Kokkos::deep_copy(execution, sharp_, host_sharp);
    for (const int m : registry.modes()) {
      PlanSet set;
      for (std::size_t component = 0; component < component_count();
           ++component) {
        const int spin = static_cast<int>(component);
        set.plans[component] = std::make_unique<coordinate_plan>(
            execution, spin, m, ell_max, theta_count);
        set.workspaces[component] =
            std::make_unique<typename coordinate_plan::Workspace>(
                *set.plans[component], radial_count,
                "plus2_linear_theta_workspace");
      }
      angular_.push_back(std::move(set));
    }
    execution.fence("initialize plus2 linear spatial workspace");
  }

  [[nodiscard]] static constexpr std::size_t component_count() {
    return static_cast<std::size_t>(Plus2MetricComponent::Count);
  }
  [[nodiscard]] metric_view stage_metric() const { return stage_; }
  [[nodiscard]] metric_view tangent_metric() const { return tangent_; }
  [[nodiscard]] metric_view second_tangent_metric() const {
    return second_tangent_;
  }
  [[nodiscard]] field_view raw_psi0() const { return raw_psi0_; }
  [[nodiscard]] field_view z_plus() const { return z_plus_; }
  [[nodiscard]] validity_view z_plus_valid() const { return z_plus_valid_; }
  [[nodiscard]] mode_view modes() const { return modes_; }
  [[nodiscard]] index_view sharp_indices() const { return sharp_; }
  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return discretization_;
  }

  template <class StageView, class TangentView, class SecondTangentView,
            class SinView, class CosView>
  void pack_reconstruction_metric(
      const execution_space& execution, const UniformRadialGrid& grid,
      const KerrParameters& parameters, const SinView& sin_theta,
      const CosView& cos_theta, const StageView& reconstruction,
      const TangentView& tangent, const SecondTangentView& second_tangent,
      const Plus2ReconstructionMetricOffsets offsets = {}) {
    validate_reconstruction_views(grid, sin_theta, cos_theta, reconstruction,
                                  tangent, second_tangent, offsets);
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count_ * radial_count * theta_count;
    using input_scalar =
        std::remove_const_t<typename StageView::value_type>;
    static_assert(std::is_same_v<input_scalar, Complex>);
    const plus2_linear_spatial_detail::PackMetricFunctor<input_scalar>
        functor{grid,
                parameters,
                sin_theta.data(),
                cos_theta.data(),
                reconstruction.data(),
                tangent.data(),
                second_tangent.data(),
                sharp_.data(),
                stage_.data(),
                tangent_.data(),
                second_tangent_.data(),
                reconstruction.extent(1),
                radial_count,
                theta_count,
                offsets.B,
                offsets.C,
                offsets.U};
    Kokkos::parallel_for(
        "pack_plus2_org_metric",
        Kokkos::RangePolicy<execution_space>(execution, 0, total), functor);
  }

  template <class SinView, class CosView>
  void evaluate_packed_metric(const execution_space& execution,
                              const UniformRadialGrid& grid,
                              const KerrParameters& parameters,
                              const SinView& sin_theta,
                              const CosView& cos_theta) {
    validate_geometry(grid, sin_theta, cos_theta);
    evaluate_radial(execution, grid);
    evaluate_angular(execution);
    evaluate_points(execution, grid, parameters, sin_theta, cos_theta);
  }

 private:
  struct PlanSet {
    std::array<std::unique_ptr<coordinate_plan>, component_count()> plans;
    std::array<std::unique_ptr<typename coordinate_plan::Workspace>,
               component_count()>
        workspaces;
  };

  template <class SinView, class CosView>
  void validate_geometry(const UniformRadialGrid& grid,
                         const SinView& sin_theta,
                         const CosView& cos_theta) const {
    if (grid.size() != radial_count_ ||
        sin_theta.extent(0) != theta_count_ ||
        cos_theta.extent(0) != theta_count_) {
      throw std::invalid_argument("plus2 linear spatial geometry mismatch");
    }
  }

  template <class SinView, class CosView, class StageView,
            class TangentView, class SecondTangentView>
  void validate_reconstruction_views(
      const UniformRadialGrid& grid, const SinView& sin_theta,
      const CosView& cos_theta, const StageView& stage,
      const TangentView& tangent, const SecondTangentView& second_tangent,
      const Plus2ReconstructionMetricOffsets offsets) const {
    validate_geometry(grid, sin_theta, cos_theta);
    const std::size_t largest =
        std::max({offsets.B, offsets.C, offsets.U});
    const auto valid = [&](const auto& view) {
      return view.extent(0) == mode_count_ && view.extent(1) > largest &&
             view.extent(2) == radial_count_ &&
             view.extent(3) == theta_count_;
    };
    if (!valid(stage) || !valid(tangent) || !valid(second_tangent)) {
      throw std::invalid_argument(
          "plus2 reconstruction metric view extent mismatch");
    }
  }

  void evaluate_radial(const execution_space& execution,
                       const UniformRadialGrid& grid) {
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const double inverse_spacing = 1.0 / grid.spacing();
    const std::size_t total =
        mode_count_ * component_count() * radial_count * theta_count;
    const plus2_linear_spatial_detail::FirstRadialFunctor first_functor{
        stage_.data(), tangent_.data(), radial_.data(), time_radial_.data(),
        radial_count, theta_count, inverse_spacing, discretization_};
    Kokkos::parallel_for(
        "plus2_linear_metric_first_radial",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        first_functor);
    const plus2_linear_spatial_detail::SecondRadialFunctor second_functor{
        radial_.data(), radial_radial_.data(), radial_count, theta_count,
        inverse_spacing, discretization_};
    Kokkos::parallel_for(
        "plus2_linear_metric_second_radial",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        second_functor);
  }

  void evaluate_angular(const execution_space& execution) {
    for (std::size_t mode = 0; mode < mode_count_; ++mode) {
      for (std::size_t component = 0; component < component_count();
           ++component) {
        auto stage = Kokkos::subview(stage_, mode, component, Kokkos::ALL,
                                     Kokkos::ALL);
        auto tangent = Kokkos::subview(tangent_, mode, component, Kokkos::ALL,
                                       Kokkos::ALL);
        auto radial = Kokkos::subview(radial_, mode, component, Kokkos::ALL,
                                      Kokkos::ALL);
        auto theta = Kokkos::subview(theta_, mode, component, Kokkos::ALL,
                                     Kokkos::ALL);
        auto theta_theta = Kokkos::subview(
            theta_theta_, mode, component, Kokkos::ALL, Kokkos::ALL);
        auto time_theta = Kokkos::subview(
            time_theta_, mode, component, Kokkos::ALL, Kokkos::ALL);
        auto radial_theta = Kokkos::subview(
            radial_theta_, mode, component, Kokkos::ALL, Kokkos::ALL);
        auto& plan = *angular_[mode].plans[component];
        auto& workspace = *angular_[mode].workspaces[component];
        plan.first(execution, stage, theta, workspace);
        plan.second(execution, stage, theta_theta, workspace);
        plan.first(execution, tangent, time_theta, workspace);
        plan.first(execution, radial, radial_theta, workspace);
      }
    }
  }

  template <class SinView, class CosView>
  void evaluate_points(const execution_space& execution,
                       const UniformRadialGrid& grid,
                       const KerrParameters& parameters,
                       const SinView& sin_theta,
                       const CosView& cos_theta) {
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const std::size_t total = mode_count_ * radial_count * theta_count;
    const plus2_linear_spatial_detail::EvaluatePointFunctor functor{
        grid,
        parameters,
        sin_theta.data(),
        cos_theta.data(),
        modes_.data(),
        stage_.data(),
        tangent_.data(),
        second_tangent_.data(),
        radial_.data(),
        time_radial_.data(),
        radial_radial_.data(),
        theta_.data(),
        time_theta_.data(),
        radial_theta_.data(),
        theta_theta_.data(),
        raw_psi0_.data(),
        z_plus_.data(),
        z_plus_valid_.data(),
        radial_count,
        theta_count};
    Kokkos::parallel_for(
        "evaluate_plus2_linear_psi0_spatial",
        Kokkos::RangePolicy<execution_space>(execution, 0, total), functor);
  }

  std::size_t mode_count_;
  std::size_t radial_count_;
  std::size_t theta_count_;
  RadialDiscretization discretization_;
  mode_view modes_;
  index_view sharp_;
  metric_view stage_;
  metric_view tangent_;
  metric_view second_tangent_;
  metric_view radial_;
  metric_view time_radial_;
  metric_view radial_radial_;
  metric_view theta_;
  metric_view time_theta_;
  metric_view radial_theta_;
  metric_view theta_theta_;
  field_view raw_psi0_;
  field_view z_plus_;
  validity_view z_plus_valid_;
  std::vector<PlanSet> angular_;
};

}  // namespace teuk
