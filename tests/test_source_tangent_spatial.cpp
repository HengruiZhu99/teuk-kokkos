#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <string>

#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/source_spatial.hpp"
#include "teuk/source_tangent_spatial.hpp"

namespace {

struct SourceInputs {
  SourceInputs(const std::string& label, const std::size_t modes,
               const std::size_t radial, const std::size_t theta)
      : fields(label + "_fields", modes,
               static_cast<std::size_t>(teuk::SpatialSourceField::Count),
               radial, theta),
        field_tangents(
            label + "_field_tangents", modes,
            static_cast<std::size_t>(teuk::SpatialSourceField::Count), radial,
            theta),
        derivatives(
            label + "_derivatives", modes,
            static_cast<std::size_t>(teuk::SpatialSourceDerivative::Count),
            radial, theta),
        derivative_tangents(
            label + "_derivative_tangents", modes,
            static_cast<std::size_t>(teuk::SpatialSourceDerivative::Count),
            radial, theta) {}

  teuk::SpatialSourceFieldView fields;
  teuk::SpatialSourceFieldView field_tangents;
  teuk::SpatialSourceDerivativeView derivatives;
  teuk::SpatialSourceDerivativeView derivative_tangents;
};

void initialize_source_inputs(SourceInputs& inputs, const std::size_t modes,
                              const std::size_t radial_points,
                              const std::size_t theta_points) {
  auto fields = Kokkos::create_mirror_view(inputs.fields);
  auto field_tangents = Kokkos::create_mirror_view(inputs.field_tangents);
  auto derivatives = Kokkos::create_mirror_view(inputs.derivatives);
  auto derivative_tangents =
      Kokkos::create_mirror_view(inputs.derivative_tangents);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t component = 0;
         component <
         static_cast<std::size_t>(teuk::SpatialSourceField::Count);
         ++component) {
      for (std::size_t radial = 0; radial < radial_points; ++radial) {
        for (std::size_t theta = 0; theta < theta_points; ++theta) {
          const double tag = 0.3 + 0.7 * mode + 0.09 * component +
                             0.013 * radial - 0.017 * theta;
          const double tangent_tag = -0.2 - 0.31 * mode + 0.04 * component -
                                     0.021 * radial + 0.008 * theta;
          fields(mode, component, radial, theta) =
              teuk::Complex(tag, -0.23 * tag + 0.05 * mode);
          field_tangents(mode, component, radial, theta) =
              teuk::Complex(tangent_tag,
                            0.37 * tangent_tag - 0.06 * component);
        }
      }
    }
    for (std::size_t component = 0;
         component <
         static_cast<std::size_t>(teuk::SpatialSourceDerivative::Count);
         ++component) {
      for (std::size_t radial = 0; radial < radial_points; ++radial) {
        for (std::size_t theta = 0; theta < theta_points; ++theta) {
          const double tag = -0.4 + 0.41 * mode - 0.07 * component +
                             0.019 * radial + 0.011 * theta;
          const double tangent_tag = 0.16 + 0.22 * mode + 0.03 * component -
                                     0.014 * radial - 0.009 * theta;
          derivatives(mode, component, radial, theta) =
              teuk::Complex(tag, 0.29 * tag - 0.02 * component);
          derivative_tangents(mode, component, radial, theta) =
              teuk::Complex(tangent_tag,
                            -0.33 * tangent_tag + 0.04 * mode);
        }
      }
    }
  }
  Kokkos::deep_copy(inputs.fields, fields);
  Kokkos::deep_copy(inputs.field_tangents, field_tangents);
  Kokkos::deep_copy(inputs.derivatives, derivatives);
  Kokkos::deep_copy(inputs.derivative_tangents, derivative_tangents);
}
template <class ExecutionSpace>
void combine_source_inputs(const ExecutionSpace& execution,
                           const SourceInputs& base, const double value_scale,
                           const double tangent_scale, SourceInputs& output) {
  const std::size_t field_total = base.fields.size();
  const std::size_t derivative_total = base.derivatives.size();
  const auto fields = base.fields;
  const auto field_tangents = base.field_tangents;
  const auto output_fields = output.fields;
  const auto output_field_tangents = output.field_tangents;
  Kokkos::parallel_for(
      "combine_source_fields",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, field_total),
      KOKKOS_LAMBDA(const std::size_t i) {
        output_fields.data()[i] = value_scale * fields.data()[i] +
                                  tangent_scale * field_tangents.data()[i];
        output_field_tangents.data()[i] = value_scale * field_tangents.data()[i];
      });
  const auto derivatives = base.derivatives;
  const auto derivative_tangents = base.derivative_tangents;
  const auto output_derivatives = output.derivatives;
  const auto output_derivative_tangents = output.derivative_tangents;
  Kokkos::parallel_for(
      "combine_source_derivatives",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, derivative_total),
      KOKKOS_LAMBDA(const std::size_t i) {
        output_derivatives.data()[i] =
            value_scale * derivatives.data()[i] +
            tangent_scale * derivative_tangents.data()[i];
        output_derivative_tangents.data()[i] =
            value_scale * derivative_tangents.data()[i];
      });
}

}  // namespace

TEST_CASE("device inner-source tangents match directional differences and scale quadratically") {
  const teuk::ModeRegistry registry({-1, 0, 1}, {-1, 1});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.7);
  constexpr std::size_t theta_points = 3;
  teuk::SpatialThetaView cos_theta("tangent_cos", theta_points);
  teuk::SpatialThetaView sin_theta("tangent_sin", theta_points);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t theta = 0; theta < theta_points; ++theta) {
    host_cos(theta) = -0.5 + 0.5 * static_cast<double>(theta);
    host_sin(theta) = std::sqrt(1.0 - host_cos(theta) * host_cos(theta));
  }
  Kokkos::deep_copy(cos_theta, host_cos);
  Kokkos::deep_copy(sin_theta, host_sin);

  SourceInputs base("tangent_base", registry.size(), grid.size(), theta_points);
  SourceInputs plus("tangent_plus", registry.size(), grid.size(), theta_points);
  SourceInputs minus("tangent_minus", registry.size(), grid.size(), theta_points);
  SourceInputs doubled("tangent_doubled", registry.size(), grid.size(),
                       theta_points);
  initialize_source_inputs(base, registry.size(), grid.size(), theta_points);
  constexpr double epsilon = 2.0e-7;
  const teuk::ExecutionSpace execution;
  combine_source_inputs(execution, base, 1.0, epsilon, plus);
  combine_source_inputs(execution, base, 1.0, -epsilon, minus);
  combine_source_inputs(execution, base, 2.0, 0.0, doubled);

  teuk::SpatialInnerSourceTangentWorkspace analytic(
      registry, grid.size(), theta_points, "analytic_tangent");
  teuk::SpatialInnerSourceTangentWorkspace scaled(
      registry, grid.size(), theta_points, "scaled_tangent");
  teuk::SpatialInnerSourceWorkspace plus_values(
      registry, grid.size(), theta_points, "plus_values");
  teuk::SpatialInnerSourceWorkspace minus_values(
      registry, grid.size(), theta_points, "minus_values");
  const teuk::KerrParameters parameters{1.0, 0.58, 1.6};
  teuk::evaluate_spatial_inner_source_tangent(
      execution, grid, parameters, cos_theta, sin_theta, base.fields,
      base.field_tangents, base.derivatives, base.derivative_tangents,
      analytic);
  teuk::evaluate_spatial_inner_source_tangent(
      execution, grid, parameters, cos_theta, sin_theta, doubled.fields,
      doubled.field_tangents, doubled.derivatives,
      doubled.derivative_tangents, scaled);
  teuk::evaluate_spatial_inner_source(execution, grid, parameters, cos_theta,
                                      sin_theta, plus.fields, plus.derivatives,
                                      plus_values);
  teuk::evaluate_spatial_inner_source(execution, grid, parameters, cos_theta,
                                      sin_theta, minus.fields,
                                      minus.derivatives, minus_values);
  execution.fence();

  const auto analytic_sum = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), analytic.summed_value());
  const auto analytic_sum_dt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), analytic.summed_tangent());
  const auto analytic_pair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), analytic.per_pair_value());
  const auto analytic_pair_dt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), analytic.per_pair_tangent());
  const auto scaled_sum = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), scaled.summed_value());
  const auto scaled_sum_dt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), scaled.summed_tangent());
  const auto scaled_pair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), scaled.per_pair_value());
  const auto scaled_pair_dt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), scaled.per_pair_tangent());
  const auto plus_sum = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), plus_values.summed());
  const auto minus_sum = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), minus_values.summed());
  const auto plus_pair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), plus_values.per_pair());
  const auto minus_pair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), minus_values.per_pair());

  for (std::size_t pair = 0; pair < registry.ordered_pairs().size(); ++pair) {
    for (std::size_t component = 0; component < 2; ++component) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < theta_points; ++theta) {
          const teuk::Complex finite_difference =
              (plus_pair(pair, component, radial, theta) -
               minus_pair(pair, component, radial, theta)) /
              (2.0 * epsilon);
          CHECK_COMPLEX_NEAR(analytic_pair_dt(pair, component, radial, theta),
                             finite_difference, 8.0e-7);
          CHECK_COMPLEX_NEAR(scaled_pair(pair, component, radial, theta),
                             4.0 * analytic_pair(pair, component, radial, theta),
                             2.0e-11);
          CHECK_COMPLEX_NEAR(
              scaled_pair_dt(pair, component, radial, theta),
              4.0 * analytic_pair_dt(pair, component, radial, theta),
              2.0e-11);
        }
      }
    }
  }
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t component = 0; component < 2; ++component) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < theta_points; ++theta) {
          const teuk::Complex finite_difference =
              (plus_sum(mode, component, radial, theta) -
               minus_sum(mode, component, radial, theta)) /
              (2.0 * epsilon);
          CHECK_COMPLEX_NEAR(analytic_sum_dt(mode, component, radial, theta),
                             finite_difference, 1.5e-6);
          CHECK_COMPLEX_NEAR(scaled_sum(mode, component, radial, theta),
                             4.0 * analytic_sum(mode, component, radial, theta),
                             4.0e-11);
          CHECK_COMPLEX_NEAR(scaled_sum_dt(mode, component, radial, theta),
                             4.0 * analytic_sum_dt(mode, component, radial, theta),
                             4.0e-11);
        }
      }
    }
  }
}

TEST_CASE("sharp inner-source tangent conjugates the negative-mode tangent") {
  const teuk::ModeRegistry registry({-1, 0, 1}, {1});
  const teuk::UniformRadialGrid grid(5, 0.0, 0.8);
  teuk::SpatialThetaView cos_theta("sharp_tangent_cos", 1);
  teuk::SpatialThetaView sin_theta("sharp_tangent_sin", 1);
  Kokkos::deep_copy(cos_theta, 0.3);
  Kokkos::deep_copy(sin_theta, std::sqrt(1.0 - 0.3 * 0.3));
  SourceInputs inputs("sharp_tangent", registry.size(), grid.size(), 1);
  Kokkos::deep_copy(inputs.fields, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(inputs.field_tangents, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(inputs.derivatives, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(inputs.derivative_tangents, teuk::Complex(0.0, 0.0));
  auto fields = Kokkos::create_mirror_view(inputs.fields);
  auto tangents = Kokkos::create_mirror_view(inputs.field_tangents);
  const std::size_t zero = registry.index(0);
  const std::size_t minus_one = registry.index(-1);
  const teuk::Complex F(0.7, -0.2);
  const teuk::Complex U(-0.4, 0.9);
  const teuk::Complex U_dt(0.3, -0.8);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    fields(zero, static_cast<std::size_t>(teuk::SpatialSourceField::F),
           radial, 0) = F;
    fields(minus_one,
           static_cast<std::size_t>(teuk::SpatialSourceField::U), radial,
           0) = U;
    tangents(minus_one,
             static_cast<std::size_t>(teuk::SpatialSourceField::U), radial,
             0) = U_dt;
  }
  Kokkos::deep_copy(inputs.fields, fields);
  Kokkos::deep_copy(inputs.field_tangents, tangents);
  teuk::SpatialInnerSourceTangentWorkspace workspace(
      registry, grid.size(), 1, "sharp_tangent_workspace");
  const teuk::KerrParameters parameters{1.0, 0.4, 1.5};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_spatial_inner_source_tangent(
      execution, grid, parameters, cos_theta, sin_theta, inputs.fields,
      inputs.field_tangents, inputs.derivatives,
      inputs.derivative_tangents, workspace);
  execution.fence();
  const auto per_pair_value = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), workspace.per_pair_value());
  const auto per_pair_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), workspace.per_pair_tangent());
  const auto [begin, end] = registry.pair_range(1);
  CHECK(end - begin == 2);
  const std::size_t pair_zero_plus = begin;
  CHECK(registry.ordered_pairs()[pair_zero_plus].m1 == 0);
  CHECK(registry.ordered_pairs()[pair_zero_plus].m2 == 1);
  constexpr std::size_t D =
      static_cast<std::size_t>(teuk::SpatialInnerSourceComponent::D);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    const double radius = grid.coordinate(radial);
    CHECK_COMPLEX_NEAR(per_pair_value(pair_zero_plus, D, radial, 0),
                       radius * F * Kokkos::conj(U), 2.0e-14);
    CHECK_COMPLEX_NEAR(per_pair_tangent(pair_zero_plus, D, radial, 0),
                       radius * F * Kokkos::conj(U_dt), 2.0e-14);
  }
}
