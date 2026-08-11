#include "test_harness.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include "teuk/modes.hpp"
#include "teuk/reconstruction_spatial.hpp"

namespace {

constexpr std::size_t reconstruction_field_count =
    static_cast<std::size_t>(teuk::ReconstructionField::Count);
constexpr std::size_t angular_input_count =
    static_cast<std::size_t>(teuk::ReconstructionAngularInput::Count);

void initialize_reconstruction_lines(
    teuk::ReconstructionRadialLines& lines,
    const teuk::ReconstructionAngularInputView& angular) {
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_state("reconstruction_host_state", lines.mode_count(),
                 reconstruction_field_count, lines.radial_point_count());
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_psi4("reconstruction_host_psi4", lines.mode_count(),
                lines.radial_point_count());
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_angular("reconstruction_host_angular", lines.mode_count(),
                   angular_input_count, lines.radial_point_count());
  for (std::size_t mode = 0; mode < lines.mode_count(); ++mode) {
    const double mode_scale = static_cast<double>(mode + 1);
    for (std::size_t radial = 0; radial < lines.radial_point_count(); ++radial) {
      const double r = lines.grid().coordinate(radial);
      const double r2 = r * r;
      const double r3 = r2 * r;
      const double r4 = r2 * r2;
      host_psi4(mode, radial) =
          teuk::Complex(mode_scale * (0.3 - 0.2 * r + 0.1 * r3),
                        -0.07 + 0.04 * mode_scale * r2);
      for (std::size_t field = 0; field < reconstruction_field_count;
           ++field) {
        const double field_scale = static_cast<double>(field + 1);
        host_state(mode, field, radial) = teuk::Complex(
            0.1 * mode_scale + 0.03 * field_scale * r +
                0.02 * (mode_scale + field_scale) * r3,
            -0.05 * field_scale + 0.04 * mode_scale * r2 -
                0.01 * field_scale * r4);
      }
      for (std::size_t input = 0; input < angular_input_count; ++input) {
        const double input_scale = static_cast<double>(input + 1);
        host_angular(mode, input, radial) = teuk::Complex(
            -0.08 * input_scale + 0.02 * mode_scale * r2,
            0.06 * mode_scale - 0.01 * input_scale * r);
      }
    }
  }
  Kokkos::deep_copy(lines.state(), host_state);
  Kokkos::deep_copy(lines.psi4(), host_psi4);
  Kokkos::deep_copy(angular, host_angular);
}

}  // namespace

TEST_CASE("SBP reconstruction evaluator agrees with all seven point equations") {
  const teuk::ModeRegistry registry({2, -2, 0});
  const teuk::UniformRadialGrid grid(13, 0.0, 0.9);
  teuk::ReconstructionRadialLines lines(registry, grid,
                                         "reconstruction_point_agreement");
  teuk::ReconstructionAngularInputView angular(
      "reconstruction_point_angular", registry.size(), angular_input_count,
      grid.size());
  initialize_reconstruction_lines(lines, angular);
  const teuk::KerrParameters parameters{1.0, 0.53, 1.8};
  constexpr double theta = 0.79;
  teuk::evaluate_sbp_reconstruction_radial_lines_rhs(lines, parameters, theta,
                                                      angular);

  const auto state =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.state());
  const auto psi4 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.psi4());
  const auto host_angular =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), angular);
  const auto rhs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.rhs());
  const auto device_radial = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.radial_derivatives());
  const auto sharp_indices = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.sharp_indices());

  constexpr std::size_t G =
      static_cast<std::size_t>(teuk::ReconstructionField::G);
  constexpr std::size_t Lambda =
      static_cast<std::size_t>(teuk::ReconstructionField::Lambda);
  constexpr std::size_t H =
      static_cast<std::size_t>(teuk::ReconstructionField::H);
  constexpr std::size_t B =
      static_cast<std::size_t>(teuk::ReconstructionField::B);
  constexpr std::size_t Pi =
      static_cast<std::size_t>(teuk::ReconstructionField::Pi);
  constexpr std::size_t C =
      static_cast<std::size_t>(teuk::ReconstructionField::C);
  constexpr std::size_t U =
      static_cast<std::size_t>(teuk::ReconstructionField::U);

  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    std::array<std::vector<teuk::Complex>, reconstruction_field_count> lines_host;
    std::array<std::vector<teuk::Complex>, reconstruction_field_count> dr_host;
    for (std::size_t field = 0; field < reconstruction_field_count; ++field) {
      lines_host[field].resize(grid.size());
      dr_host[field].resize(grid.size());
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        lines_host[field][radial] = state(mode, field, radial);
      }
      teuk::d42_first_derivative(grid, lines_host[field], dr_host[field]);
    }

    const std::size_t sharp_mode = sharp_indices(mode);
    CHECK(sharp_mode == registry.sharp_index(registry.modes()[mode]));
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (std::size_t field = 0; field < reconstruction_field_count; ++field) {
        CHECK_COMPLEX_NEAR(device_radial(mode, field, radial),
                           dr_host[field][radial], 3.0e-13);
      }
      const double radius = grid.coordinate(radial);
      const auto background =
          teuk::kerr_background_point(parameters, radius, theta);
      const teuk::ReconstructionFields point_fields{
          psi4(mode, radial),
          state(mode, G, radial),
          state(mode, H, radial),
          state(mode, Lambda, radial),
          state(mode, Pi, radial),
          state(mode, B, radial),
          state(mode, C, radial),
          state(mode, U, radial),
          Kokkos::conj(state(sharp_mode, Pi, radial)),
          Kokkos::conj(state(sharp_mode, B, radial)),
          Kokkos::conj(state(sharp_mode, C, radial))};
      const teuk::ReconstructionAngularDerivatives point_angular{
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::Eth1F),
              radial),
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::Eth2G),
              radial),
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::Eth2C),
              radial),
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::Eth2Pi),
              radial),
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::EthPrime1BSharp),
              radial),
          host_angular(
              mode,
              static_cast<std::size_t>(
                  teuk::ReconstructionAngularInput::EthPrime2CSharp),
              radial)};
      const auto delta = teuk::reconstruction_delta_rhs(
          radius, background, point_fields, point_angular);
      const std::array<teuk::Complex, reconstruction_field_count> values{
          point_fields.G, point_fields.Lambda, point_fields.H, point_fields.B,
          point_fields.Pi, point_fields.C, point_fields.U};
      const std::array<teuk::Complex, reconstruction_field_count> sources{
          delta.G, delta.Lambda, delta.H, delta.B, delta.Pi, delta.C, delta.U};
      const std::array<int, reconstruction_field_count> falloffs{2, 1, 3, 1,
                                                                 2, 2, 3};
      for (std::size_t field = 0; field < reconstruction_field_count; ++field) {
        const teuk::Complex expected = teuk::reconstruction_time_derivative(
            values[field], dr_host[field][radial], sources[field],
            falloffs[field], radius, parameters.mass,
            parameters.compactification_length);
        CHECK_COMPLEX_NEAR(rhs(mode, field, radial), expected, 5.0e-13);
      }
    }
  }
}

TEST_CASE("reconstruction sharp terms read conjugate negative signed mode") {
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  teuk::ReconstructionRadialLines lines(registry, grid,
                                         "reconstruction_sharp_lookup");
  teuk::ReconstructionAngularInputView angular(
      "reconstruction_sharp_angular", registry.size(), angular_input_count,
      grid.size());
  Kokkos::deep_copy(lines.state(), teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(lines.psi4(), teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(angular, teuk::Complex(0.0, 0.0));

  auto host_state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.state());
  constexpr std::size_t Pi =
      static_cast<std::size_t>(teuk::ReconstructionField::Pi);
  constexpr std::size_t U =
      static_cast<std::size_t>(teuk::ReconstructionField::U);
  const std::size_t plus = registry.index(1);
  const std::size_t minus = registry.index(-1);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    host_state(plus, Pi, radial) = teuk::Complex(2.0, 3.0);
  }
  Kokkos::deep_copy(lines.state(), host_state);

  const teuk::KerrParameters parameters{1.0, 0.67, 1.6};
  constexpr double theta = 0.88;
  teuk::evaluate_sbp_reconstruction_radial_lines_rhs(lines, parameters, theta,
                                                      angular);
  const auto rhs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.rhs());
  const auto sharp_indices = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.sharp_indices());
  CHECK(sharp_indices(minus) == plus);
  CHECK(sharp_indices(plus) == minus);

  for (std::size_t radial = 1; radial < grid.size(); ++radial) {
    const double radius = grid.coordinate(radial);
    const auto background =
        teuk::kerr_background_point(parameters, radius, theta);
    const teuk::Complex delta_u_minus =
        -2.0 * radius * background.pi0 * teuk::Complex(2.0, -3.0);
    const teuk::Complex delta_u_plus =
        -2.0 * radius * Kokkos::conj(background.pi0) *
        teuk::Complex(2.0, 3.0);
    const teuk::Complex expected_minus = teuk::reconstruction_time_derivative(
        teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0), delta_u_minus, 3,
        radius, parameters.mass, parameters.compactification_length);
    const teuk::Complex expected_plus = teuk::reconstruction_time_derivative(
        teuk::Complex(0.0, 0.0), teuk::Complex(0.0, 0.0), delta_u_plus, 3,
        radius, parameters.mass, parameters.compactification_length);
    CHECK_COMPLEX_NEAR(rhs(minus, U, radial), expected_minus, 3.0e-13);
    CHECK_COMPLEX_NEAR(rhs(plus, U, radial), expected_plus, 3.0e-13);
    CHECK(Kokkos::abs(rhs(minus, U, radial) - rhs(plus, U, radial)) > 1.0e-5);
  }
}

TEST_CASE("reconstruction spatial storage rejects missing sharp modes") {
  bool threw = false;
  try {
    const teuk::ModeRegistry incomplete({0, 2});
    const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
    const teuk::ReconstructionRadialLines lines(incomplete, grid,
                                                 "missing_sharp_mode");
    static_cast<void>(lines);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("reconstruction spatial evaluator validates angular stage extents") {
  const teuk::ModeRegistry registry({0});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  teuk::ReconstructionRadialLines lines(registry, grid,
                                         "bad_angular_extents");
  teuk::ReconstructionAngularInputView angular(
      "bad_angular_input", registry.size(), angular_input_count - 1,
      grid.size());
  bool threw = false;
  try {
    teuk::evaluate_sbp_reconstruction_radial_lines_rhs(
        lines, teuk::KerrParameters{}, 0.7, angular);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("caller RK stage views execute directly on the active backend") {
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(11, 0.0, 0.8);
  teuk::ReconstructionRadialLines owned(registry, grid,
                                         "external_stage_owner");
  const teuk::Complex sentinel(91.0, -37.0);
  Kokkos::deep_copy(owned.state(), sentinel);

  teuk::ReconstructionRadialStateView stage_a(
      "external_stage_a", registry.size(), reconstruction_field_count,
      grid.size());
  teuk::ReconstructionRadialStateView stage_b(
      "external_stage_b", registry.size(), reconstruction_field_count,
      grid.size());
  teuk::ReconstructionRadialValueView psi4_a("external_psi4_a",
                                              registry.size(), grid.size());
  teuk::ReconstructionRadialValueView psi4_b("external_psi4_b",
                                              registry.size(), grid.size());
  teuk::ReconstructionAngularInputView angular_a(
      "external_angular_a", registry.size(), angular_input_count, grid.size());
  teuk::ReconstructionAngularInputView angular_b(
      "external_angular_b", registry.size(), angular_input_count, grid.size());
  teuk::ReconstructionRadialStateView rhs_a(
      "external_rhs_a", registry.size(), reconstruction_field_count,
      grid.size());
  teuk::ReconstructionRadialStateView rhs_b(
      "external_rhs_b", registry.size(), reconstruction_field_count,
      grid.size());
  teuk::ReconstructionRadialStateView dr_a(
      "external_dr_a", registry.size(), reconstruction_field_count,
      grid.size());
  teuk::ReconstructionRadialStateView dr_b(
      "external_dr_b", registry.size(), reconstruction_field_count,
      grid.size());

  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_stage_a("external_host_stage_a", registry.size(),
                   reconstruction_field_count, grid.size());
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_psi4_a("external_host_psi4_a", registry.size(), grid.size());
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_angular_a("external_host_angular_a", registry.size(),
                     angular_input_count, grid.size());
  constexpr double stage_scale = 1.6;
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const double r = grid.coordinate(radial);
      host_psi4_a(mode, radial) =
          teuk::Complex(0.2 * (mode + 1) + 0.1 * r * r, -0.03 + 0.07 * r);
      for (std::size_t field = 0; field < reconstruction_field_count;
           ++field) {
        host_stage_a(mode, field, radial) = teuk::Complex(
            0.04 * (mode + 1) * (field + 1) + 0.02 * r * r * r,
            -0.03 * (field + 1) + 0.01 * (mode + 1) * r * r);
      }
      for (std::size_t input = 0; input < angular_input_count; ++input) {
        host_angular_a(mode, input, radial) = teuk::Complex(
            -0.02 * (input + 1) + 0.03 * (mode + 1) * r,
            0.01 * (mode + input + 2) - 0.015 * r * r);
      }
    }
  }
  Kokkos::deep_copy(stage_a, host_stage_a);
  Kokkos::deep_copy(psi4_a, host_psi4_a);
  Kokkos::deep_copy(angular_a, host_angular_a);
  teuk::ExecutionSpace execution;
  Kokkos::parallel_for(
      "form_external_reconstruction_stage_b",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(
          execution, 0,
          registry.size() * reconstruction_field_count * grid.size()),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode_field = flat / grid.size();
        const std::size_t radial = flat - mode_field * grid.size();
        const std::size_t mode = mode_field / reconstruction_field_count;
        const std::size_t field =
            mode_field - mode * reconstruction_field_count;
        stage_b(mode, field, radial) =
            stage_scale * stage_a(mode, field, radial);
      });
  Kokkos::parallel_for(
      "form_external_reconstruction_inputs_b",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(
          execution, 0, registry.size() * grid.size()),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t mode = flat / grid.size();
        const std::size_t radial = flat - mode * grid.size();
        psi4_b(mode, radial) = stage_scale * psi4_a(mode, radial);
        for (std::size_t input = 0; input < angular_input_count; ++input) {
          angular_b(mode, input, radial) =
              stage_scale * angular_a(mode, input, radial);
        }
      });

  const teuk::KerrParameters parameters{1.0, 0.48, 1.75};
  constexpr double theta = 0.82;
  teuk::evaluate_sbp_reconstruction_radial_views_rhs(
      execution, grid, parameters, theta, owned.sharp_indices(), stage_a,
      psi4_a, angular_a, rhs_a, dr_a);
  teuk::evaluate_sbp_reconstruction_radial_views_rhs(
      execution, grid, parameters, theta, owned.sharp_indices(), stage_b,
      psi4_b, angular_b, rhs_b, dr_b);
  execution.fence();

  const auto host_rhs_a =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs_a);
  const auto host_rhs_b =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs_b);
  const auto host_dr_a =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dr_a);
  const auto host_dr_b =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dr_b);
  const auto owned_state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), owned.state());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < reconstruction_field_count; ++field) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        CHECK_COMPLEX_NEAR(owned_state(mode, field, radial), sentinel, 0.0);
        CHECK_COMPLEX_NEAR(host_dr_b(mode, field, radial),
                           stage_scale * host_dr_a(mode, field, radial),
                           5.0e-13);
        CHECK_COMPLEX_NEAR(host_rhs_b(mode, field, radial),
                           stage_scale * host_rhs_a(mode, field, radial),
                           8.0e-13);
      }
    }
  }

  // Independently assemble one signed-mode point from stage A.
  const std::size_t mode = registry.index(1);
  const std::size_t sharp_mode = registry.index(-1);
  const std::size_t radial = 6;
  constexpr std::size_t G =
      static_cast<std::size_t>(teuk::ReconstructionField::G);
  constexpr std::size_t Lambda =
      static_cast<std::size_t>(teuk::ReconstructionField::Lambda);
  constexpr std::size_t H =
      static_cast<std::size_t>(teuk::ReconstructionField::H);
  constexpr std::size_t B =
      static_cast<std::size_t>(teuk::ReconstructionField::B);
  constexpr std::size_t Pi =
      static_cast<std::size_t>(teuk::ReconstructionField::Pi);
  constexpr std::size_t C =
      static_cast<std::size_t>(teuk::ReconstructionField::C);
  constexpr std::size_t U =
      static_cast<std::size_t>(teuk::ReconstructionField::U);
  const double radius = grid.coordinate(radial);
  const teuk::ReconstructionFields fields{
      host_psi4_a(mode, radial),
      host_stage_a(mode, G, radial),
      host_stage_a(mode, H, radial),
      host_stage_a(mode, Lambda, radial),
      host_stage_a(mode, Pi, radial),
      host_stage_a(mode, B, radial),
      host_stage_a(mode, C, radial),
      host_stage_a(mode, U, radial),
      Kokkos::conj(host_stage_a(sharp_mode, Pi, radial)),
      Kokkos::conj(host_stage_a(sharp_mode, B, radial)),
      Kokkos::conj(host_stage_a(sharp_mode, C, radial))};
  const teuk::ReconstructionAngularDerivatives angular_point{
      host_angular_a(mode, 0, radial), host_angular_a(mode, 1, radial),
      host_angular_a(mode, 2, radial), host_angular_a(mode, 3, radial),
      host_angular_a(mode, 4, radial), host_angular_a(mode, 5, radial)};
  const auto delta = teuk::reconstruction_delta_rhs(
      radius, teuk::kerr_background_point(parameters, radius, theta), fields,
      angular_point);
  const std::array<teuk::Complex, reconstruction_field_count> values{
      fields.G, fields.Lambda, fields.H, fields.B, fields.Pi, fields.C,
      fields.U};
  const std::array<teuk::Complex, reconstruction_field_count> sources{
      delta.G, delta.Lambda, delta.H, delta.B, delta.Pi, delta.C, delta.U};
  const std::array<int, reconstruction_field_count> falloffs{2, 1, 3, 1, 2, 2,
                                                              3};
  for (std::size_t field = 0; field < reconstruction_field_count; ++field) {
    const teuk::Complex expected = teuk::reconstruction_time_derivative(
        values[field], host_dr_a(mode, field, radial), sources[field],
        falloffs[field], radius, parameters.mass,
        parameters.compactification_length);
    CHECK_COMPLEX_NEAR(host_rhs_a(mode, field, radial), expected, 5.0e-13);
  }
}
