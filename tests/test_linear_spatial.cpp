#include "test_harness.hpp"

#include <array>
#include <vector>

#include "teuk/linear_spatial.hpp"

namespace {

void initialize_lines(teuk::TeukolskyRadialLines& lines) {
  auto host_state = Kokkos::create_mirror_view(lines.state());
  auto host_angular = Kokkos::create_mirror_view(lines.angular_laplacian());
  auto host_forcing = Kokkos::create_mirror_view(lines.forcing());
  for (std::size_t mode = 0; mode < lines.mode_count(); ++mode) {
    for (std::size_t radial = 0; radial < lines.radial_point_count(); ++radial) {
      const double r = lines.grid().coordinate(radial);
      const double scale = static_cast<double>(mode + 1);
      host_state(mode, static_cast<std::size_t>(teuk::TeukolskyField::P),
                 radial) = teuk::Complex(scale * (0.2 + r * r), -0.1 * r);
      host_state(mode, static_cast<std::size_t>(teuk::TeukolskyField::Q),
                 radial) = teuk::Complex(scale * (-0.3 + r + r * r * r),
                                         0.2 * r * r);
      host_state(mode, static_cast<std::size_t>(teuk::TeukolskyField::Psi),
                 radial) = teuk::Complex(scale * (0.4 - r + r * r * r * r),
                                         -0.15 * r * r * r);
      host_angular(mode, radial) =
          teuk::Complex(-0.5 * scale * r, 0.07 + 0.03 * r);
      host_forcing(mode, radial) =
          teuk::Complex(0.11 - 0.02 * r, 0.04 * scale);
    }
  }
  Kokkos::deep_copy(lines.state(), host_state);
  Kokkos::deep_copy(lines.angular_laplacian(), host_angular);
  Kokkos::deep_copy(lines.forcing(), host_forcing);
}

}  // namespace

TEST_CASE("SBP spatial evaluator matches compact host RHS for signed modes") {
  const teuk::ModeRegistry registry({2, -2, 0});
  const teuk::UniformRadialGrid grid(13, 0.0, 1.1);
  teuk::TeukolskyRadialLines lines(registry, grid, "signed_mode_rhs");
  initialize_lines(lines);

  teuk::TeukolskyParameters base;
  base.mass = 1.0;
  base.spin = 0.43;
  base.compactification_length = 1.8;
  base.spin_weight = -2;
  base.reduction_damping = 0.37;
  constexpr double theta = 0.83;
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      lines, base, theta, teuk::ReductionEvolution::FreeDamped);
  const auto state =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.state());
  const auto angular = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.angular_laplacian());
  const auto forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), lines.forcing());
  const auto rhs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lines.rhs());

  constexpr std::size_t p = static_cast<std::size_t>(teuk::TeukolskyField::P);
  constexpr std::size_t q = static_cast<std::size_t>(teuk::TeukolskyField::Q);
  constexpr std::size_t psi =
      static_cast<std::size_t>(teuk::TeukolskyField::Psi);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    std::vector<teuk::Complex> p_line(grid.size());
    std::vector<teuk::Complex> q_line(grid.size());
    std::vector<teuk::Complex> psi_line(grid.size());
    std::vector<teuk::Complex> dr_q(grid.size());
    std::vector<teuk::Complex> dr_psi(grid.size());
    std::vector<teuk::Complex> velocity(grid.size());
    std::vector<teuk::Complex> dr_velocity(grid.size());
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      p_line[radial] = state(mode, p, radial);
      q_line[radial] = state(mode, q, radial);
      psi_line[radial] = state(mode, psi, radial);
    }
    teuk::d42_first_derivative(grid, q_line, dr_q);
    teuk::d42_first_derivative(grid, psi_line, dr_psi);
    teuk::TeukolskyParameters parameters = base;
    parameters.azimuthal_mode = registry.modes()[mode];
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const auto coefficients = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), theta);
      velocity[radial] = teuk::teukolsky_psi_rhs(
          coefficients,
          teuk::TeukolskyState{p_line[radial], q_line[radial],
                               psi_line[radial]});
    }
    teuk::d42_first_derivative(grid, velocity, dr_velocity);
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const auto coefficients = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), theta);
      const teuk::TeukolskyState point{p_line[radial], q_line[radial],
                                       psi_line[radial]};
      CHECK_COMPLEX_NEAR(
          rhs(mode, p, radial),
          teuk::teukolsky_p_rhs(coefficients, point, dr_q[radial],
                                angular(mode, radial), forcing(mode, radial)),
          3.0e-13);
      CHECK_COMPLEX_NEAR(
          rhs(mode, q, radial),
          teuk::teukolsky_q_rhs(
              dr_velocity[radial], q_line[radial] - dr_psi[radial],
              base.reduction_damping),
          3.0e-13);
      CHECK_COMPLEX_NEAR(rhs(mode, psi, radial), velocity[radial], 3.0e-13);
    }
  }
}

TEST_CASE("SBP spatial free reduction damping targets only the constraint") {
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(12, 0.0, 1.0);
  teuk::TeukolskyRadialLines undamped(registry, grid, "undamped_constraint");
  teuk::TeukolskyRadialLines damped(registry, grid, "damped_constraint");
  auto host = Kokkos::create_mirror_view(undamped.state());
  constexpr std::size_t p = static_cast<std::size_t>(teuk::TeukolskyField::P);
  constexpr std::size_t q = static_cast<std::size_t>(teuk::TeukolskyField::Q);
  constexpr std::size_t psi =
      static_cast<std::size_t>(teuk::TeukolskyField::Psi);
  const std::array<teuk::Complex, 2> offsets{
      teuk::Complex(0.3, -0.2), teuk::Complex(-0.1, 0.4)};
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      const double r = grid.coordinate(radial);
      host(mode, p, radial) = teuk::Complex(0.1 * r, -0.2);
      host(mode, psi, radial) = teuk::Complex(r * r, -0.5 * r);
      host(mode, q, radial) = teuk::Complex(2.0 * r, -0.5) + offsets[mode];
    }
  }
  Kokkos::deep_copy(undamped.state(), host);
  Kokkos::deep_copy(damped.state(), host);
  teuk::TeukolskyParameters zero_gamma;
  zero_gamma.spin = 0.2;
  zero_gamma.compactification_length = 2.0;
  teuk::TeukolskyParameters positive_gamma = zero_gamma;
  positive_gamma.reduction_damping = 1.6;
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      undamped, zero_gamma, 0.7, teuk::ReductionEvolution::FreeDamped);
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      damped, positive_gamma, 0.7, teuk::ReductionEvolution::FreeDamped);
  const auto rhs_zero = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), undamped.rhs());
  const auto rhs_positive = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), damped.rhs());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      CHECK_COMPLEX_NEAR(rhs_positive(mode, p, radial),
                         rhs_zero(mode, p, radial), 3.0e-13);
      CHECK_COMPLEX_NEAR(rhs_positive(mode, psi, radial),
                         rhs_zero(mode, psi, radial), 3.0e-13);
      CHECK_COMPLEX_NEAR(rhs_positive(mode, q, radial) -
                             rhs_zero(mode, q, radial),
                         -positive_gamma.reduction_damping * offsets[mode],
                         2.0e-11);
    }
  }
}

TEST_CASE("SBP stage-constrained spatial evolution ignores inconsistent Q") {
  const teuk::ModeRegistry registry({0});
  const teuk::UniformRadialGrid grid(14, 0.0, 1.0);
  teuk::TeukolskyRadialLines consistent(registry, grid, "consistent_q");
  teuk::TeukolskyRadialLines inconsistent(registry, grid, "inconsistent_q");
  // Allocate independent host staging explicitly. With the Serial backend,
  // create_mirror_view may alias an already-host-resident source View; mutating
  // that mirror after the first deep_copy would then corrupt `consistent`.
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace> host(
      "stage_constraint_host", registry.size(),
      static_cast<std::size_t>(teuk::TeukolskyField::Count), grid.size());
  constexpr std::size_t p = static_cast<std::size_t>(teuk::TeukolskyField::P);
  constexpr std::size_t q = static_cast<std::size_t>(teuk::TeukolskyField::Q);
  constexpr std::size_t psi =
      static_cast<std::size_t>(teuk::TeukolskyField::Psi);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    const double r = grid.coordinate(radial);
    host(0, p, radial) = teuk::Complex(0.2 + r, -0.1 * r);
    host(0, psi, radial) = teuk::Complex(r * r, 0.2 * r);
    host(0, q, radial) = teuk::Complex(2.0 * r, 0.2);
  }
  Kokkos::deep_copy(consistent.state(), host);
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    host(0, q, radial) += teuk::Complex(5.0, -2.0);
  }
  Kokkos::deep_copy(inconsistent.state(), host);
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.35;
  parameters.compactification_length = 1.9;
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      consistent, parameters, 0.8, teuk::ReductionEvolution::FreeDamped);
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      inconsistent, parameters, 0.8,
      teuk::ReductionEvolution::StageConstrained);
  const auto free_rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), consistent.rhs());
  const auto constrained_rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), inconsistent.rhs());
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    CHECK_COMPLEX_NEAR(constrained_rhs(0, p, radial), free_rhs(0, p, radial),
                       2.0e-11);
    CHECK_COMPLEX_NEAR(constrained_rhs(0, q, radial), free_rhs(0, q, radial),
                       2.0e-11);
    CHECK_COMPLEX_NEAR(constrained_rhs(0, psi, radial),
                       free_rhs(0, psi, radial), 2.0e-11);
  }
}

TEST_CASE("SBP spatial evaluator adds compatible dissipation inside the RHS") {
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(15, 0.0, 1.0);
  teuk::TeukolskyRadialLines plain(registry, grid, "plain_rhs");
  teuk::TeukolskyRadialLines dissipative(registry, grid, "dissipative_rhs");
  initialize_lines(plain);
  Kokkos::deep_copy(dissipative.state(), plain.state());
  Kokkos::deep_copy(dissipative.angular_laplacian(), plain.angular_laplacian());
  Kokkos::deep_copy(dissipative.forcing(), plain.forcing());
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.25;
  parameters.compactification_length = 1.8;
  constexpr double strength = 0.03;
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      plain, parameters, 0.6, teuk::ReductionEvolution::FreeDamped, 0.0);
  teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
      dissipative, parameters, 0.6, teuk::ReductionEvolution::FreeDamped,
      strength);
  const auto state =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plain.state());
  const auto plain_rhs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plain.rhs());
  const auto dissipative_rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), dissipative.rhs());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0;
         field < static_cast<std::size_t>(teuk::TeukolskyField::Count);
         ++field) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        const teuk::Complex expected = teuk::d42_compatible_dissipation_at(
            &state(mode, field, 0), grid.size(), radial, grid.spacing(),
            strength);
        CHECK_COMPLEX_NEAR(dissipative_rhs(mode, field, radial) -
                               plain_rhs(mode, field, radial),
                           expected, 3.0e-12);
      }
    }
  }
}
