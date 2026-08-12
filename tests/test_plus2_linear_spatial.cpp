#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/boundary.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_linear_spatial.hpp"
#include "teuk/types.hpp"

namespace {

using C = teuk::Complex;
using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space = execution_space::memory_space;
using state_view =
    Kokkos::View<C****, Kokkos::LayoutRight, memory_space>;
using host_state =
    Kokkos::View<C****, Kokkos::LayoutRight, Kokkos::HostSpace>;

constexpr std::size_t component(const teuk::Plus2MetricComponent value) {
  return static_cast<std::size_t>(value);
}

double raise_factor(const int ell, const int spin) {
  return std::abs(spin) > ell ? 0.0
                              : teuk::angular::raising_factor(ell, spin);
}

double lower_factor(const int ell, const int spin) {
  return std::abs(spin) > ell ? 0.0
                              : teuk::angular::lowering_factor(ell, spin);
}

double harmonic(const int ell, const int m, const int spin,
                const double theta) {
  if (std::abs(spin) > ell || std::abs(m) > ell) return 0.0;
  return teuk::angular::spin_weighted_harmonic_theta(ell, m, spin, theta);
}

// Independent host transcription of ordinary-coordinate theta derivatives.
// It deliberately does not call DeviceSpinCoordinateDerivativePlan.
std::array<double, 3> harmonic_coordinate_jet(const int ell, const int m,
                                               const int spin,
                                               const double theta) {
  const double up = raise_factor(ell, spin);
  const double down = lower_factor(ell, spin);
  const double value = harmonic(ell, m, spin, theta);
  const double first =
      -0.5 * (up * harmonic(ell, m, spin + 1, theta) +
              down * harmonic(ell, m, spin - 1, theta));
  const double second =
      0.25 *
      (up * raise_factor(ell, spin + 1) *
           harmonic(ell, m, spin + 2, theta) +
       up * lower_factor(ell, spin + 1) * value +
       down * raise_factor(ell, spin - 1) * value +
       down * lower_factor(ell, spin - 1) *
           harmonic(ell, m, spin - 2, theta));
  return {value, first, second};
}

std::array<C, 3> signed_amplitudes(const int m) {
  const double sign = m < 0 ? -1.0 : 1.0;
  return {C(0.31 + 0.04 * sign, -0.17 + 0.03 * sign),
          C(-0.23 + 0.02 * sign, 0.29 + 0.05 * sign),
          C(0.19 - 0.06 * sign, 0.11 + 0.04 * sign)};
}

std::array<C, 3> signed_rates(const int m) {
  const double sign = m < 0 ? -1.0 : 1.0;
  return {C(0.13, -0.07 * sign), C(-0.09, 0.11 * sign),
          C(0.06, 0.08 * sign)};
}

std::array<int, 3> manufactured_ells() { return {3, 4, 3}; }

std::array<double, 3> polynomial_radial_jet(const std::size_t c,
                                             const double radius) {
  if (c == component(teuk::Plus2MetricComponent::LL)) {
    return {radius * radius, 2.0 * radius, 2.0};
  }
  if (c == component(teuk::Plus2MetricComponent::LM)) {
    return {0.8 * radius * radius, 1.6 * radius, 1.6};
  }
  return {radius * (1.0 + 0.35 * radius), 1.0 + 0.7 * radius, 0.7};
}

teuk::Plus2OrgMetricFields load_fields(
    const std::array<C, 3>& values) {
  return {values[component(teuk::Plus2MetricComponent::LL)],
          values[component(teuk::Plus2MetricComponent::LM)],
          values[component(teuk::Plus2MetricComponent::MM)]};
}

teuk::Plus2LinearPsi0Point manufactured_point(
    const teuk::KerrParameters& parameters, const double radius,
    const double theta, const int m) {
  const auto amplitudes = signed_amplitudes(m);
  const auto rates = signed_rates(m);
  const auto ells = manufactured_ells();
  std::array<C, 3> h{};
  std::array<C, 3> h_t{};
  std::array<C, 3> h_tt{};
  std::array<C, 3> h_r{};
  std::array<C, 3> h_tr{};
  std::array<C, 3> h_rr{};
  std::array<C, 3> h_theta{};
  std::array<C, 3> h_ttheta{};
  std::array<C, 3> h_rtheta{};
  std::array<C, 3> h_thetatheta{};
  for (std::size_t c = 0; c < h.size(); ++c) {
    const auto angular = harmonic_coordinate_jet(
        ells[c], m, static_cast<int>(c), theta);
    const auto radial = polynomial_radial_jet(c, radius);
    h[c] = amplitudes[c] * radial[0] * angular[0];
    h_t[c] = rates[c] * h[c];
    h_tt[c] = rates[c] * rates[c] * h[c];
    h_r[c] = amplitudes[c] * radial[1] * angular[0];
    h_tr[c] = rates[c] * h_r[c];
    h_rr[c] = amplitudes[c] * radial[2] * angular[0];
    h_theta[c] = amplitudes[c] * radial[0] * angular[1];
    h_ttheta[c] = rates[c] * h_theta[c];
    h_rtheta[c] = amplitudes[c] * radial[1] * angular[1];
    h_thetatheta[c] = amplitudes[c] * radial[0] * angular[2];
  }
  teuk::Plus2OrgMetricStage stage{load_fields(h), load_fields(h_t),
                                   load_fields(h_tt)};
  teuk::Plus2OrgMetricDerivativeSlots derivatives{};
  derivatives.h_R = load_fields(h_r);
  derivatives.h_TR = load_fields(h_tr);
  derivatives.h_RR = load_fields(h_rr);
  derivatives.h_theta = load_fields(h_theta);
  derivatives.h_Ttheta = load_fields(h_ttheta);
  derivatives.h_Rtheta = load_fields(h_rtheta);
  derivatives.h_thetatheta = load_fields(h_thetatheta);
  teuk::plus2_fill_modal_azimuthal_derivatives(m, stage, derivatives);
  return teuk::evaluate_plus2_linear_psi0(
      parameters, radius, std::sin(theta), std::cos(theta), stage,
      derivatives);
}

struct Angles {
  Kokkos::View<double*, memory_space> sine;
  Kokkos::View<double*, memory_space> cosine;
  std::vector<double> theta;

  Angles(const execution_space& execution, const int count)
      : sine("plus2_linear_sine", count),
        cosine("plus2_linear_cosine", count) {
    const auto grid = teuk::angular::gauss_legendre(count);
    auto host_sine = Kokkos::create_mirror_view(sine);
    auto host_cosine = Kokkos::create_mirror_view(cosine);
    theta.resize(static_cast<std::size_t>(count));
    for (int node = 0; node < count; ++node) {
      const std::size_t i = static_cast<std::size_t>(node);
      host_cosine(i) = grid.x[i];
      host_sine(i) = std::sqrt(1.0 - grid.x[i] * grid.x[i]);
      theta[i] = std::acos(grid.x[i]);
    }
    Kokkos::deep_copy(execution, sine, host_sine);
    Kokkos::deep_copy(execution, cosine, host_cosine);
  }
};

void fill_reconstruction_for_manufactured_metric(
    const execution_space& execution, const teuk::ModeRegistry& registry,
    const teuk::UniformRadialGrid& grid,
    const teuk::KerrParameters& parameters, const Angles& angles,
    const state_view& stage, const state_view& tangent,
    const state_view& second) {
  host_state host_stage("plus2_linear_host_stage", stage.extent(0),
                        stage.extent(1), stage.extent(2), stage.extent(3));
  host_state host_tangent("plus2_linear_host_tangent", tangent.extent(0),
                          tangent.extent(1), tangent.extent(2),
                          tangent.extent(3));
  host_state host_second("plus2_linear_host_second", second.extent(0),
                         second.extent(1), second.extent(2),
                         second.extent(3));
  for (std::size_t input_mode = 0; input_mode < registry.size();
       ++input_mode) {
    const int m = registry.modes()[input_mode];
    const int sharp_output_m = -m;
    const auto direct_amplitudes = signed_amplitudes(m);
    const auto direct_rates = signed_rates(m);
    const auto sharp_amplitudes = signed_amplitudes(sharp_output_m);
    const auto sharp_rates = signed_rates(sharp_output_m);
    const auto ells = manufactured_ells();
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const double radius = grid.coordinate(radial);
      for (std::size_t node = 0; node < angles.theta.size(); ++node) {
        const auto background = teuk::kerr_background_point(
            parameters, radius, std::cos(angles.theta[node]),
            std::sin(angles.theta[node]));
        const double y_ll =
            harmonic(ells[0], m, 0, angles.theta[node]);
        const C u = background.mu0 * direct_amplitudes[0] * y_ll;
        host_stage(input_mode, 2, radial, node) = u;
        host_tangent(input_mode, 2, radial, node) = direct_rates[0] * u;
        host_second(input_mode, 2, radial, node) =
            direct_rates[0] * direct_rates[0] * u;

        const double y_lm = harmonic(ells[1], sharp_output_m, 1,
                                     angles.theta[node]);
        const C c_field =
            Kokkos::conj(0.8 * sharp_amplitudes[1] * y_lm);
        host_stage(input_mode, 1, radial, node) = c_field;
        host_tangent(input_mode, 1, radial, node) =
            Kokkos::conj(sharp_rates[1]) * c_field;
        host_second(input_mode, 1, radial, node) =
            Kokkos::conj(sharp_rates[1] * sharp_rates[1]) * c_field;

        const double y_mm = harmonic(ells[2], sharp_output_m, 2,
                                     angles.theta[node]);
        const C b_field = Kokkos::conj(
            (1.0 + 0.35 * radius) * sharp_amplitudes[2] * y_mm);
        host_stage(input_mode, 0, radial, node) = b_field;
        host_tangent(input_mode, 0, radial, node) =
            Kokkos::conj(sharp_rates[2]) * b_field;
        host_second(input_mode, 0, radial, node) =
            Kokkos::conj(sharp_rates[2] * sharp_rates[2]) * b_field;
      }
    }
  }
  Kokkos::deep_copy(execution, stage, host_stage);
  Kokkos::deep_copy(execution, tangent, host_tangent);
  Kokkos::deep_copy(execution, second, host_second);
}

int linear_spatial_allocations = 0;
int linear_spatial_deep_copies = 0;
int linear_spatial_fences = 0;

void count_linear_spatial_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                     const void*, std::uint64_t) {
  ++linear_spatial_allocations;
}

void count_linear_spatial_deep_copy(Kokkos::Tools::SpaceHandle, const char*,
                                    const void*, Kokkos::Tools::SpaceHandle,
                                    const char*, const void*, std::uint64_t) {
  ++linear_spatial_deep_copies;
}

void count_linear_spatial_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++linear_spatial_fences;
}

struct GraphErrors {
  double boundary = 0.0;
  double interior = 0.0;
  double global = 0.0;
};

GraphErrors smooth_graph_errors(
    const std::size_t radial_count,
    const teuk::RadialDiscretization discretization =
        teuk::RadialDiscretization::D42) {
  const execution_space execution;
  const teuk::ModeRegistry registry({0});
  const teuk::KerrParameters parameters{1.0, 0.999, 1.7};
  const teuk::TeukolskyParameters teuk_parameters{
      parameters.mass, parameters.spin, parameters.compactification_length};
  const double horizon =
      teuk::compactified_outer_horizon_radius(teuk_parameters);
  const teuk::UniformRadialGrid grid(radial_count, 0.0, horizon);
  constexpr int theta_count = 10;
  constexpr int ell = 3;
  const Angles angles(execution, theta_count);
  teuk::Plus2LinearPsi0SpatialWorkspace<execution_space> workspace(
      execution, registry, radial_count, ell, theta_count,
      "plus2_linear_convergence", discretization);
  auto stage = workspace.stage_metric();
  auto tangent = workspace.tangent_metric();
  auto second = workspace.second_tangent_metric();
  auto host_stage = Kokkos::create_mirror_view(stage);
  auto host_tangent = Kokkos::create_mirror_view(tangent);
  auto host_second = Kokkos::create_mirror_view(second);
  const std::array<C, 3> amplitudes{C(0.21, -0.13), C(-0.17, 0.19),
                                     C(0.12, 0.23)};
  const std::array<C, 3> rates{C(0.07, -0.02), C(-0.04, 0.05),
                                C(0.03, 0.06)};
  const std::array<double, 3> frequencies{2.1, 1.7, 2.4};
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double radius = grid.coordinate(radial);
    for (int node = 0; node < theta_count; ++node) {
      for (std::size_t c = 0; c < 3; ++c) {
        const double y = harmonic(ell, 0, static_cast<int>(c),
                                  angles.theta[static_cast<std::size_t>(node)]);
        const double g = 0.7 + std::sin(frequencies[c] * radius);
        const C value = amplitudes[c] * g * y;
        host_stage(0, c, radial, node) = value;
        host_tangent(0, c, radial, node) = rates[c] * value;
        host_second(0, c, radial, node) = rates[c] * rates[c] * value;
      }
    }
  }
  Kokkos::deep_copy(execution, stage, host_stage);
  Kokkos::deep_copy(execution, tangent, host_tangent);
  Kokkos::deep_copy(execution, second, host_second);
  workspace.evaluate_packed_metric(execution, grid, parameters, angles.sine,
                                   angles.cosine);
  execution.fence("finish plus2 linear convergence graph");
  const auto actual = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.raw_psi0());
  GraphErrors errors;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double radius = grid.coordinate(radial);
    for (int node = 0; node < theta_count; ++node) {
      std::array<C, 3> h{};
      std::array<C, 3> h_t{};
      std::array<C, 3> h_tt{};
      std::array<C, 3> h_r{};
      std::array<C, 3> h_tr{};
      std::array<C, 3> h_rr{};
      std::array<C, 3> h_theta{};
      std::array<C, 3> h_ttheta{};
      std::array<C, 3> h_rtheta{};
      std::array<C, 3> h_thetatheta{};
      const double theta = angles.theta[static_cast<std::size_t>(node)];
      for (std::size_t c = 0; c < 3; ++c) {
        const auto angular = harmonic_coordinate_jet(
            ell, 0, static_cast<int>(c), theta);
        const double k = frequencies[c];
        const double g = 0.7 + std::sin(k * radius);
        const double gp = k * std::cos(k * radius);
        const double gpp = -k * k * std::sin(k * radius);
        h[c] = amplitudes[c] * g * angular[0];
        h_t[c] = rates[c] * h[c];
        h_tt[c] = rates[c] * rates[c] * h[c];
        h_r[c] = amplitudes[c] * gp * angular[0];
        h_tr[c] = rates[c] * h_r[c];
        h_rr[c] = amplitudes[c] * gpp * angular[0];
        h_theta[c] = amplitudes[c] * g * angular[1];
        h_ttheta[c] = rates[c] * h_theta[c];
        h_rtheta[c] = amplitudes[c] * gp * angular[1];
        h_thetatheta[c] = amplitudes[c] * g * angular[2];
      }
      teuk::Plus2OrgMetricStage exact_stage{
          load_fields(h), load_fields(h_t), load_fields(h_tt)};
      teuk::Plus2OrgMetricDerivativeSlots exact_derivatives{};
      exact_derivatives.h_R = load_fields(h_r);
      exact_derivatives.h_TR = load_fields(h_tr);
      exact_derivatives.h_RR = load_fields(h_rr);
      exact_derivatives.h_theta = load_fields(h_theta);
      exact_derivatives.h_Ttheta = load_fields(h_ttheta);
      exact_derivatives.h_Rtheta = load_fields(h_rtheta);
      exact_derivatives.h_thetatheta = load_fields(h_thetatheta);
      teuk::plus2_fill_modal_azimuthal_derivatives(
          0, exact_stage, exact_derivatives);
      const auto expected = teuk::evaluate_plus2_linear_psi0(
          parameters, radius, std::sin(theta), std::cos(theta), exact_stage,
          exact_derivatives);
      const double error = Kokkos::abs(
          actual(0, radial, static_cast<std::size_t>(node)) -
          expected.psi0_code_tetrad);
      errors.global = std::max(errors.global, error);
      if (radial < 4 || radial + 4 >= radial_count) {
        errors.boundary = std::max(errors.boundary, error);
      }
      if (radial >= 8 && radial + 8 < radial_count) {
        errors.interior = std::max(errors.interior, error);
      }
    }
  }
  return errors;
}

}  // namespace

TEST_CASE("plus2 linear spatial graph is zero preserving and scri fail closed") {
  const execution_space execution;
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr int theta_count = 8;
  const Angles angles(execution, theta_count);
  state_view stage("plus2_linear_zero_stage", registry.size(), 3, grid.size(),
                   theta_count);
  state_view tangent("plus2_linear_zero_tangent", registry.size(), 3,
                     grid.size(), theta_count);
  state_view second("plus2_linear_zero_second", registry.size(), 3,
                    grid.size(), theta_count);
  Kokkos::deep_copy(execution, stage, C{});
  Kokkos::deep_copy(execution, tangent, C{});
  Kokkos::deep_copy(execution, second, C{});
  teuk::Plus2LinearPsi0SpatialWorkspace<execution_space> workspace(
      execution, registry, grid.size(), 3, theta_count);
  const teuk::KerrParameters parameters{1.0, 0.61, 1.4};
  workspace.pack_reconstruction_metric(execution, grid, parameters,
                                       angles.sine, angles.cosine, stage,
                                       tangent, second);
  workspace.evaluate_packed_metric(execution, grid, parameters, angles.sine,
                                   angles.cosine);
  execution.fence("finish zero plus2 linear spatial test");
  const auto raw = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.raw_psi0());
  const auto zplus = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus());
  const auto valid = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus_valid());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (int theta = 0; theta < theta_count; ++theta) {
        CHECK(raw(mode, radial, static_cast<std::size_t>(theta)) == C{});
        CHECK(zplus(mode, radial, static_cast<std::size_t>(theta)) == C{});
        CHECK(valid(mode, radial, static_cast<std::size_t>(theta)) ==
              static_cast<std::uint8_t>(radial == 0 ? 0 : 1));
      }
    }
  }
}

TEST_CASE("plus2 linear spatial graph matches nonzero signed Kerr oracle") {
  const execution_space execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::KerrParameters parameters{1.0, 0.999, 1.8};
  const teuk::TeukolskyParameters teuk_parameters{
      parameters.mass, parameters.spin, parameters.compactification_length};
  const double horizon =
      teuk::compactified_outer_horizon_radius(teuk_parameters);
  const teuk::UniformRadialGrid grid(17, 0.0, horizon);
  constexpr int theta_count = 14;
  constexpr int ell_max = 5;
  const Angles angles(execution, theta_count);
  state_view stage("plus2_linear_oracle_stage", registry.size(), 3,
                   grid.size(), theta_count);
  state_view tangent("plus2_linear_oracle_tangent", registry.size(), 3,
                     grid.size(), theta_count);
  state_view second("plus2_linear_oracle_second", registry.size(), 3,
                    grid.size(), theta_count);
  fill_reconstruction_for_manufactured_metric(
      execution, registry, grid, parameters, angles, stage, tangent, second);
  teuk::Plus2LinearPsi0SpatialWorkspace<execution_space> workspace(
      execution, registry, grid.size(), ell_max, theta_count,
      "plus2_linear_oracle");
  workspace.pack_reconstruction_metric(execution, grid, parameters,
                                       angles.sine, angles.cosine, stage,
                                       tangent, second);
  workspace.evaluate_packed_metric(execution, grid, parameters, angles.sine,
                                   angles.cosine);
  execution.fence("finish nonzero plus2 linear spatial oracle");
  const auto raw = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.raw_psi0());
  const auto zplus = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus());
  const auto valid = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus_valid());
  const auto packed_stage = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.stage_metric());
  const auto packed_tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.tangent_metric());
  const auto packed_second = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.second_tangent_metric());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    const auto amplitudes = signed_amplitudes(m);
    const auto rates = signed_rates(m);
    const auto ells = manufactured_ells();
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const double radius = grid.coordinate(radial);
      for (int node = 0; node < theta_count; ++node) {
        const std::size_t theta_index = static_cast<std::size_t>(node);
        const double theta = angles.theta[theta_index];
        for (std::size_t c = 0; c < 3; ++c) {
          const double y = harmonic(ells[c], m, static_cast<int>(c), theta);
          const C expected_metric =
              amplitudes[c] * polynomial_radial_jet(c, radius)[0] * y;
          CHECK_COMPLEX_NEAR(packed_stage(mode, c, radial, theta_index),
                             expected_metric, 4.0e-15);
          CHECK_COMPLEX_NEAR(packed_tangent(mode, c, radial, theta_index),
                             rates[c] * expected_metric, 4.0e-15);
          CHECK_COMPLEX_NEAR(
              packed_second(mode, c, radial, theta_index),
              rates[c] * rates[c] * expected_metric, 4.0e-15);
        }
        const auto expected =
            manufactured_point(parameters, radius, theta, m);
        CHECK_COMPLEX_NEAR(raw(mode, radial, theta_index),
                           expected.psi0_code_tetrad, 2.0e-11);
        if (radial == 0) {
          CHECK(valid(mode, radial, theta_index) == 0);
          CHECK(zplus(mode, radial, theta_index) == C{});
        } else {
          CHECK(valid(mode, radial, theta_index) == 1);
          const auto expected_regularized = teuk::regularize_plus2_linear_psi0(
              expected.psi0_code_tetrad, radius, std::cos(theta),
              parameters.spin, parameters.compactification_length);
          CHECK(expected_regularized.valid);
          CHECK_COMPLEX_NEAR(zplus(mode, radial, theta_index),
                             expected_regularized.z_plus, 2.0e-5);
        }
        if (radial + 1 == grid.size()) {
          CHECK(std::isfinite(raw(mode, radial, theta_index).real()));
          CHECK(std::isfinite(raw(mode, radial, theta_index).imag()));
          CHECK(std::isfinite(zplus(mode, radial, theta_index).real()));
          CHECK(std::isfinite(zplus(mode, radial, theta_index).imag()));
        }
      }
    }
  }
}

TEST_CASE("plus2 linear spatial graph resolves the endpoint order limitation") {
  const auto error_33 = smooth_graph_errors(33);
  const auto error_65 = smooth_graph_errors(65);
  const auto error_129 = smooth_graph_errors(129);
  const auto error_257 = smooth_graph_errors(257);
  CHECK(error_33.interior > 0.0);
  CHECK(error_65.interior > 0.0);
  CHECK(error_129.interior > 0.0);
  CHECK(error_33.interior / error_65.interior > 12.0);
  CHECK(error_65.interior / error_129.interior > 12.0);
  // D4-2 supplies fourth-order centered rows, but only a second-order first-
  // derivative closure.  Composing it for h_RR approaches a factor-eight
  // endpoint rate here, not the factor sixteen required for a global
  // fourth-order claim.
  CHECK(error_33.boundary / error_65.boundary > 7.0);
  CHECK(error_65.boundary / error_129.boundary > 7.0);
  CHECK(error_129.boundary / error_257.boundary > 7.0);
  CHECK(error_33.boundary / error_65.boundary < 15.0);
  CHECK(error_65.boundary / error_129.boundary < 15.0);
  CHECK(error_129.boundary / error_257.boundary < 15.0);
  CHECK(error_33.global / error_65.global < 15.0);
  CHECK(error_65.global / error_129.global < 15.0);
  CHECK(error_129.global / error_257.global < 15.0);
}

TEST_CASE("plus2 linear spatial D8-4 graph is globally fourth order") {
  const auto error_33 = smooth_graph_errors(
      33, teuk::RadialDiscretization::D84);
  const auto error_65 = smooth_graph_errors(
      65, teuk::RadialDiscretization::D84);
  const auto error_129 = smooth_graph_errors(
      129, teuk::RadialDiscretization::D84);
  CHECK(error_33.boundary / error_65.boundary > 12.0);
  CHECK(error_65.boundary / error_129.boundary > 12.0);
  CHECK(error_33.global / error_65.global > 12.0);
  CHECK(error_65.global / error_129.global > 12.0);
}

TEST_CASE("plus2 linear spatial stage launches allocate copy and fence nothing") {
  const execution_space execution;
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr int theta_count = 8;
  const Angles angles(execution, theta_count);
  const teuk::KerrParameters parameters{1.0, 0.61, 1.4};
  state_view stage("plus2_linear_noalloc_stage", registry.size(), 3,
                   grid.size(), theta_count);
  state_view tangent("plus2_linear_noalloc_tangent", registry.size(), 3,
                     grid.size(), theta_count);
  state_view second("plus2_linear_noalloc_second", registry.size(), 3,
                    grid.size(), theta_count);
  Kokkos::deep_copy(stage, C(0.1, -0.2));
  Kokkos::deep_copy(tangent, C(-0.03, 0.04));
  Kokkos::deep_copy(second, C(0.02, 0.01));
  teuk::Plus2LinearPsi0SpatialWorkspace<execution_space> workspace(
      execution, registry, grid.size(), 3, theta_count,
      "plus2_linear_noalloc");

  linear_spatial_allocations = 0;
  linear_spatial_deep_copies = 0;
  linear_spatial_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_linear_spatial_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_linear_spatial_deep_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_linear_spatial_fence);
  {
    Kokkos::View<C*> probe("plus2_linear_callback_probe", 1);
    Kokkos::deep_copy(probe, C{});
    execution.fence("plus2 linear callback positive control");
  }
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(linear_spatial_allocations > 0);
  CHECK(linear_spatial_deep_copies > 0);
  CHECK(linear_spatial_fences > 0);

  workspace.pack_reconstruction_metric(execution, grid, parameters,
                                       angles.sine, angles.cosine, stage,
                                       tangent, second);
  workspace.evaluate_packed_metric(execution, grid, parameters, angles.sine,
                                   angles.cosine);
  execution.fence("warm plus2 linear launch storage");
  linear_spatial_allocations = 0;
  linear_spatial_deep_copies = 0;
  linear_spatial_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_linear_spatial_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_linear_spatial_deep_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_linear_spatial_fence);
  workspace.pack_reconstruction_metric(execution, grid, parameters,
                                       angles.sine, angles.cosine, stage,
                                       tangent, second);
  workspace.evaluate_packed_metric(execution, grid, parameters, angles.sine,
                                   angles.cosine);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish plus2 linear no-allocation stage");
  CHECK(linear_spatial_allocations == 0);
  CHECK(linear_spatial_deep_copies == 0);
  CHECK(linear_spatial_fences == 0);
}
