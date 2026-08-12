#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "teuk/angular.hpp"
#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/second_order.hpp"
#include "teuk/source_spatial.hpp"

TEST_CASE("device spatial source preserves deterministic pairs and sharp lookup") {
  const teuk::ModeRegistry registry({-2, 0, 2}, {-2, 0, 2});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const auto theta_grid = teuk::angular::gauss_legendre(7);
  teuk::SpatialThetaView cos_theta("source_cos_theta", theta_grid.size());
  teuk::SpatialThetaView sin_theta("source_sin_theta", theta_grid.size());
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t theta = 0; theta < theta_grid.size(); ++theta) {
    host_cos(theta) = theta_grid.x[theta];
    host_sin(theta) =
        std::sqrt(1.0 - theta_grid.x[theta] * theta_grid.x[theta]);
  }
  Kokkos::deep_copy(cos_theta, host_cos);
  Kokkos::deep_copy(sin_theta, host_sin);

  teuk::SpatialSourceFieldView fields(
      "source_fields", registry.size(),
      static_cast<std::size_t>(teuk::SpatialSourceField::Count),
      radial_grid.size(), theta_grid.size());
  teuk::SpatialSourceDerivativeView derivatives(
      "source_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::SpatialSourceDerivative::Count),
      radial_grid.size(), theta_grid.size());
  auto host_fields = Kokkos::create_mirror_view(fields);
  auto host_derivatives = Kokkos::create_mirror_view(derivatives);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t component = 0;
         component < static_cast<std::size_t>(teuk::SpatialSourceField::Count);
         ++component) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < theta_grid.size(); ++theta) {
          const double tag = 1.0 + 3.0 * mode + 0.2 * component +
                             0.03 * radial + 0.004 * theta;
          host_fields(mode, component, radial, theta) =
              teuk::Complex(tag, -0.37 * tag + 0.1 * mode);
        }
      }
    }
    for (std::size_t component = 0;
         component <
         static_cast<std::size_t>(teuk::SpatialSourceDerivative::Count);
         ++component) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < theta_grid.size(); ++theta) {
          const double tag = -0.4 + 2.0 * mode - 0.13 * component +
                             0.02 * radial - 0.005 * theta;
          host_derivatives(mode, component, radial, theta) =
              teuk::Complex(tag, 0.23 * tag - 0.07 * component);
        }
      }
    }
  }
  Kokkos::deep_copy(fields, host_fields);
  Kokkos::deep_copy(derivatives, host_derivatives);
  teuk::SpatialInnerSourceWorkspace workspace(
      registry, radial_grid.size(), theta_grid.size(), "source_test");
  const teuk::KerrParameters parameters{1.0, 0.62, 1.7};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_spatial_inner_source(execution, radial_grid, parameters,
                                      cos_theta, sin_theta, fields,
                                      derivatives, workspace);
  execution.fence();

  const auto summed = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.summed());
  const auto per_pair = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.per_pair());
  const auto sharp = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.sharp_indices());
  const std::size_t radial = 4;
  const std::size_t theta = 3;
  for (const int target_m : registry.targets()) {
    const auto [begin, end] = registry.pair_range(target_m);
    teuk::Complex expected_D(0.0, 0.0);
    teuk::Complex expected_T(0.0, 0.0);
    for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
      const teuk::ModePair& pair = registry.ordered_pairs()[pair_index];
      const std::size_t m1 = pair.index1;
      const std::size_t m2 = pair.index2;
      const auto field = [&](const std::size_t mode,
                             const teuk::SpatialSourceField component) {
        return host_fields(mode, static_cast<std::size_t>(component), radial,
                           theta);
      };
      const auto derivative = [&] (
          const teuk::SpatialSourceDerivative component) {
        return host_derivatives(m2, static_cast<std::size_t>(component),
                                radial, theta);
      };
      const teuk::OrderedPairFields point_fields{
          field(m1, teuk::SpatialSourceField::F),
          field(m1, teuk::SpatialSourceField::G),
          field(m1, teuk::SpatialSourceField::Lambda),
          field(m1, teuk::SpatialSourceField::Pi),
          field(m1, teuk::SpatialSourceField::B),
          field(m1, teuk::SpatialSourceField::C),
          field(m1, teuk::SpatialSourceField::U),
          field(m2, teuk::SpatialSourceField::F),
          field(m2, teuk::SpatialSourceField::G),
          field(m2, teuk::SpatialSourceField::H),
          field(m2, teuk::SpatialSourceField::B),
          field(m2, teuk::SpatialSourceField::C),
          field(m2, teuk::SpatialSourceField::U),
          Kokkos::conj(field(sharp(m2), teuk::SpatialSourceField::U)),
          Kokkos::conj(field(sharp(m2), teuk::SpatialSourceField::C)),
          Kokkos::conj(field(sharp(m1), teuk::SpatialSourceField::C)),
          Kokkos::conj(field(sharp(m1), teuk::SpatialSourceField::B)),
          Kokkos::conj(field(sharp(m2), teuk::SpatialSourceField::Pi)),
          Kokkos::conj(field(sharp(m2), teuk::SpatialSourceField::B))};
      const teuk::OrderedPairDerivatives point_derivatives{
          derivative(teuk::SpatialSourceDerivative::Delta1F),
          derivative(teuk::SpatialSourceDerivative::Delta3U),
          derivative(teuk::SpatialSourceDerivative::Eth2C),
          derivative(teuk::SpatialSourceDerivative::EthPrime2CSharp),
          derivative(teuk::SpatialSourceDerivative::Eth1B),
          derivative(teuk::SpatialSourceDerivative::Delta2C),
          derivative(teuk::SpatialSourceDerivative::Delta2G),
          derivative(teuk::SpatialSourceDerivative::Eth2G),
          derivative(teuk::SpatialSourceDerivative::EthPrime1F),
          derivative(teuk::SpatialSourceDerivative::Delta2CSharp),
          derivative(teuk::SpatialSourceDerivative::EthPrime1BSharp)};
      const auto background = teuk::kerr_background_point(
          parameters, radial_grid.coordinate(radial), host_cos(theta),
          host_sin(theta));
      const auto expected = teuk::corrected_ordered_pair_source(
          radial_grid.coordinate(radial), background, point_fields,
          point_derivatives);
      CHECK_COMPLEX_NEAR(
          per_pair(pair_index,
                   static_cast<std::size_t>(
                       teuk::SpatialInnerSourceComponent::D),
                   radial, theta),
          expected.D, 2.0e-12);
      CHECK_COMPLEX_NEAR(
          per_pair(pair_index,
                   static_cast<std::size_t>(
                       teuk::SpatialInnerSourceComponent::T),
                   radial, theta),
          expected.T, 2.0e-12);
      expected_D += expected.D;
      expected_T += expected.T;
    }
    const std::size_t target = registry.index(target_m);
    CHECK_COMPLEX_NEAR(
        summed(target,
               static_cast<std::size_t>(
                   teuk::SpatialInnerSourceComponent::D),
               radial, theta),
        expected_D, 3.0e-12);
    CHECK_COMPLEX_NEAR(
        summed(target,
               static_cast<std::size_t>(
                   teuk::SpatialInnerSourceComponent::T),
               radial, theta),
        expected_T, 3.0e-12);
  }

  const std::size_t plus = registry.index(2);
  CHECK(sharp(plus) == registry.index(-2));
  CHECK(Kokkos::abs(host_fields(sharp(plus), 0, radial, theta) -
                    host_fields(plus, 0, radial, theta)) > 1.0);
}

TEST_CASE("spatial source rejects modes without sharp closure") {
  bool rejected = false;
  try {
    const teuk::ModeRegistry registry({0, 2});
    const teuk::SpatialInnerSourceWorkspace workspace(registry, 9, 5);
    static_cast<void>(workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("device spatial outer source includes radial angular and tangent terms") {
  const teuk::UniformRadialGrid radial_grid(17, 0.0, 0.9);
  const auto theta_grid = teuk::angular::gauss_legendre(6);
  constexpr std::size_t mode_count = 3;
  teuk::SpatialThetaView cos_theta("outer_cos", theta_grid.size());
  teuk::SpatialThetaView sin_theta("outer_sin", theta_grid.size());
  teuk::SpatialInnerSourceView inner(
      "outer_inner", mode_count,
      static_cast<std::size_t>(teuk::SpatialInnerSourceComponent::Count),
      radial_grid.size(), theta_grid.size());
  teuk::SpatialInnerSourceView inner_dt(
      "outer_inner_dt", mode_count,
      static_cast<std::size_t>(teuk::SpatialInnerSourceComponent::Count),
      radial_grid.size(), theta_grid.size());
  teuk::SpatialOuterSourceView lowered("outer_lowered", mode_count,
                                       radial_grid.size(), theta_grid.size());
  teuk::SpatialOuterSourceView source("outer_source", mode_count,
                                      radial_grid.size(), theta_grid.size());
  teuk::SpatialOuterSourceView forcing("outer_forcing", mode_count,
                                       radial_grid.size(), theta_grid.size());
  teuk::SpatialOuterSourceView ethprime("outer_ethprime", mode_count,
                                        radial_grid.size(), theta_grid.size());
  teuk::SpatialOuterSourceView source_from_ethprime(
      "outer_source_precomputed", mode_count, radial_grid.size(),
      theta_grid.size());
  teuk::SpatialOuterSourceView forcing_from_ethprime(
      "outer_forcing_precomputed", mode_count, radial_grid.size(),
      theta_grid.size());
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  auto host_inner = Kokkos::create_mirror_view(inner);
  auto host_dt = Kokkos::create_mirror_view(inner_dt);
  auto host_lowered = Kokkos::create_mirror_view(lowered);
  auto host_ethprime = Kokkos::create_mirror_view(ethprime);
  for (std::size_t theta = 0; theta < theta_grid.size(); ++theta) {
    host_cos(theta) = theta_grid.x[theta];
    host_sin(theta) =
        std::sqrt(1.0 - theta_grid.x[theta] * theta_grid.x[theta]);
  }
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      const double r = radial_grid.coordinate(radial);
      for (std::size_t theta = 0; theta < theta_grid.size(); ++theta) {
        const double x = theta_grid.x[theta];
        host_inner(mode, 0, radial, theta) = teuk::Complex(
            (mode + 1) * (0.2 + r * r + 0.3 * r * r * r), 0.1 * x * r);
        host_inner(mode, 1, radial, theta) =
            teuk::Complex(-0.3 + 0.2 * mode + r * x, 0.07 * r * r);
        host_dt(mode, 0, radial, theta) =
            teuk::Complex(0.11 + 0.03 * r, -0.04 * x);
        host_dt(mode, 1, radial, theta) =
            teuk::Complex(-0.08 + 0.02 * mode, 0.05 * r * x);
        host_lowered(mode, radial, theta) =
            teuk::Complex(0.17 * (mode + 1) * x, -0.09 + 0.01 * r);
        host_ethprime(mode, radial, theta) = teuk::ethprime_n_point(
            host_inner(mode, 1, radial, theta),
            host_dt(mode, 1, radial, theta),
            host_lowered(mode, radial, theta), -1, -2, r,
            host_sin(theta), host_cos(theta), 0.58, 1.6);
      }
    }
  }
  Kokkos::deep_copy(cos_theta, host_cos);
  Kokkos::deep_copy(sin_theta, host_sin);
  Kokkos::deep_copy(inner, host_inner);
  Kokkos::deep_copy(inner_dt, host_dt);
  Kokkos::deep_copy(lowered, host_lowered);
  Kokkos::deep_copy(ethprime, host_ethprime);
  const teuk::KerrParameters parameters{1.0, 0.58, 1.6};
  const teuk::ExecutionSpace execution;
  teuk::evaluate_spatial_outer_source(
      execution, radial_grid, parameters, cos_theta, sin_theta, inner,
      inner_dt, lowered, source, forcing, teuk::RadialDiscretization::D84);
  teuk::evaluate_spatial_outer_source_from_ethprime(
      execution, radial_grid, parameters, cos_theta, sin_theta, inner,
      inner_dt, ethprime, source_from_ethprime, forcing_from_ethprime,
      teuk::RadialDiscretization::D84);
  execution.fence();
  const auto host_source = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, source);
  const auto host_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, forcing);
  const auto host_source_from_ethprime = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, source_from_ethprime);
  const auto host_forcing_from_ethprime = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, forcing_from_ethprime);

  const std::size_t mode = 2;
  const std::size_t theta = 4;
  std::vector<teuk::Complex> D_line(radial_grid.size());
  for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
    D_line[radial] = host_inner(mode, 0, radial, theta);
  }
  for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
    const teuk::Complex dr_D = teuk::radial_first_derivative_at(
        teuk::RadialDiscretization::D84, D_line.data(), radial_grid.size(),
        radial, 1.0 / radial_grid.spacing());
    const double radius = radial_grid.coordinate(radial);
    const auto background = teuk::kerr_background_point(
        parameters, radius, host_cos(theta), host_sin(theta));
    const teuk::Complex delta_D = teuk::delta_n_point(
        host_inner(mode, 0, radial, theta),
        host_dt(mode, 0, radial, theta), dr_D, 3, radius,
        parameters.mass, parameters.compactification_length);
    const teuk::Complex ethprime_T = teuk::ethprime_n_point(
        host_inner(mode, 1, radial, theta),
        host_dt(mode, 1, radial, theta),
        host_lowered(mode, radial, theta), -1, -2, radius, host_sin(theta),
        host_cos(theta), parameters.spin,
        parameters.compactification_length);
    const teuk::Complex expected = teuk::outer_source_over_r3(
        radius, background,
        teuk::InnerSource{host_inner(mode, 0, radial, theta),
                          host_inner(mode, 1, radial, theta)},
        teuk::OuterSourceDerivatives{delta_D, ethprime_T});
    CHECK_COMPLEX_NEAR(host_source(mode, radial, theta), expected, 2.0e-12);
    CHECK_COMPLEX_NEAR(host_source_from_ethprime(mode, radial, theta),
                       expected, 2.0e-12);
    // Test the source algebra above and the coordinate normalization here as
    // two independent operations.  Comparing the device forcing directly to
    // a fully host-reassembled source compounds the source roundoff with the
    // O(10) normalization factor and makes an absolute-only threshold depend
    // on backend evaluation order.  The two checks together imply the same
    // composed formula without amplifying the accepted source discrepancy.
    CHECK_COMPLEX_NEAR(
        host_forcing(mode, radial, theta),
        teuk::coordinate_second_order_forcing(
            radius, host_cos(theta), parameters.spin,
            parameters.compactification_length,
            host_source(mode, radial, theta)),
        2.0e-12);
    CHECK_COMPLEX_NEAR(host_forcing_from_ethprime(mode, radial, theta),
                       host_forcing(mode, radial, theta), 2.0e-12);
  }
}
