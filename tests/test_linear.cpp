#include "test_harness.hpp"

#include <cmath>
#include <vector>

#include "teuk/grid.hpp"
#include "teuk/teukolsky.hpp"

TEST_CASE("Schwarzschild Teukolsky coefficients have the analytic scri limit") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.0;
  parameters.compactification_length = 2.0;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = 3;

  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, 0.0, 0.73);
  CHECK_NEAR(coefficients.time, 16.0, 1.0e-14);
  CHECK_NEAR(coefficients.radial_advection, 4.0, 1.0e-14);
  CHECK_NEAR(coefficients.radial_principal, 0.0, 1.0e-14);
  CHECK_COMPLEX_NEAR(coefficients.definition, teuk::Complex(8.0, 0.0),
                     1.0e-14);
  CHECK_COMPLEX_NEAR(coefficients.q, teuk::Complex(0.0, 0.0), 1.0e-14);
  CHECK_COMPLEX_NEAR(coefficients.psi, teuk::Complex(0.0, 0.0), 1.0e-14);
}

TEST_CASE("Teukolsky radial principal coefficient vanishes at both ends") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.6;
  parameters.compactification_length = 2.0;
  const double boyer_lindquist_horizon =
      parameters.mass +
      std::sqrt(parameters.mass * parameters.mass -
                parameters.spin * parameters.spin);
  const double radial_horizon =
      parameters.compactification_length *
      parameters.compactification_length / boyer_lindquist_horizon;

  CHECK_NEAR(teuk::teukolsky_coefficients(parameters, 0.0, 0.4)
                 .radial_principal,
             0.0, 1.0e-14);
  CHECK_NEAR(teuk::teukolsky_coefficients(parameters, radial_horizon, 0.4)
                 .radial_principal,
             0.0, 2.0e-14);
}

TEST_CASE("rotating Teukolsky coefficients match an independent point oracle") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.4;
  parameters.compactification_length = 1.7;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -3;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, 0.31, 0.82);

  CHECK_NEAR(coefficients.time, 19.1802374515822, 2.0e-14);
  CHECK_NEAR(coefficients.radial_advection, 2.6315824618958104, 2.0e-14);
  CHECK_NEAR(coefficients.radial_principal, 0.07566030861699452, 2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.definition,
      teuk::Complex(7.943583050969218, -4.52131171712731), 2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.q,
      teuk::Complex(-0.6842223823948469, 0.07980622837370245), 2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.psi,
      teuk::Complex(0.21821482022485367, 0.25743944636678207), 2.0e-14);
}

TEST_CASE("compact point RHS implements P Q psi equations") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.4;
  parameters.compactification_length = 1.7;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -3;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, 0.31, 0.82);
  const teuk::TeukolskyState state{teuk::Complex(0.2, -0.7),
                                   teuk::Complex(-0.4, 0.3),
                                   teuk::Complex(0.8, -0.1)};
  const teuk::Complex dq(0.6, -0.2);
  const teuk::Complex angular(-1.1, 0.4);
  const teuk::Complex forcing(0.9, 0.7);

  const teuk::Complex expected_psi =
      (state.P + 2.0 * coefficients.radial_advection * state.Q -
       coefficients.definition * state.psi) /
      coefficients.time;
  const teuk::Complex expected_p =
      coefficients.radial_principal * dq + coefficients.q * state.Q +
      coefficients.psi * state.psi + angular + forcing;
  CHECK_COMPLEX_NEAR(teuk::teukolsky_psi_rhs(coefficients, state),
                     expected_psi, 2.0e-14);
  CHECK_COMPLEX_NEAR(
      teuk::teukolsky_p_rhs(coefficients, state, dq, angular, forcing),
      expected_p, 2.0e-14);
  CHECK_COMPLEX_NEAR(
      teuk::teukolsky_q_rhs(teuk::Complex(0.3, -0.2),
                            teuk::Complex(-0.4, 0.7), 1.5),
      teuk::Complex(0.9, -1.25), 2.0e-14);
}

TEST_CASE("free reduction damping acts only on Q minus radial derivative psi") {
  const teuk::UniformRadialGrid grid(17, 0.0, 1.0);
  teuk::TeukolskyParameters undamped;
  undamped.spin = 0.2;
  undamped.compactification_length = 2.0;
  undamped.azimuthal_mode = 1;
  teuk::TeukolskyParameters damped = undamped;
  damped.reduction_damping = 1.75;

  std::vector<teuk::TeukolskyState> state(grid.size());
  std::vector<teuk::Complex> angular(grid.size(), 0.0);
  std::vector<teuk::Complex> forcing(grid.size(), 0.0);
  std::vector<teuk::TeukolskyState> rhs_undamped(grid.size());
  std::vector<teuk::TeukolskyState> rhs_damped(grid.size());
  teuk::TeukolskyRadialWorkspace workspace_undamped(grid.size());
  teuk::TeukolskyRadialWorkspace workspace_damped(grid.size());
  const teuk::Complex offset(0.3, -0.2);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.coordinate(i);
    state[i].P = teuk::Complex(0.2 * r, -0.1);
    state[i].psi = teuk::Complex(r * r * r * r, -0.5 * r * r);
    state[i].Q = teuk::Complex(4.0 * r * r * r, -r) + offset;
  }

  teuk::evaluate_teukolsky_radial_line_rhs(
      grid, undamped, 0.7, state, angular, forcing,
      teuk::ReductionEvolution::FreeDamped, workspace_undamped,
      rhs_undamped);
  teuk::evaluate_teukolsky_radial_line_rhs(
      grid, damped, 0.7, state, angular, forcing,
      teuk::ReductionEvolution::FreeDamped, workspace_damped, rhs_damped);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK_COMPLEX_NEAR(rhs_damped[i].P, rhs_undamped[i].P, 2.0e-13);
    CHECK_COMPLEX_NEAR(rhs_damped[i].psi, rhs_undamped[i].psi, 2.0e-13);
    CHECK_COMPLEX_NEAR(rhs_damped[i].Q - rhs_undamped[i].Q,
                       -damped.reduction_damping * offset, 2.0e-11);
  }
}

TEST_CASE("stage constrained and consistent free reductions agree") {
  const teuk::UniformRadialGrid grid(15, 0.0, 1.0);
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.3;
  parameters.compactification_length = 2.0;
  parameters.azimuthal_mode = -2;
  std::vector<teuk::TeukolskyState> consistent(grid.size());
  std::vector<teuk::TeukolskyState> unconstrained(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.coordinate(i);
    consistent[i].P = teuk::Complex(0.1 + r, -0.2 * r);
    consistent[i].psi = teuk::Complex(r * r * r, 0.25 * r * r * r * r);
    consistent[i].Q = teuk::Complex(3.0 * r * r, r * r * r);
    unconstrained[i] = consistent[i];
    unconstrained[i].Q += teuk::Complex(10.0 - r, 4.0);
  }
  std::vector<teuk::Complex> angular(grid.size(), teuk::Complex(-0.4, 0.1));
  std::vector<teuk::Complex> forcing(grid.size(), teuk::Complex(0.2, -0.3));
  std::vector<teuk::TeukolskyState> rhs_free(grid.size());
  std::vector<teuk::TeukolskyState> rhs_constrained(grid.size());
  teuk::TeukolskyRadialWorkspace free_workspace(grid.size());
  teuk::TeukolskyRadialWorkspace constrained_workspace(grid.size());

  teuk::evaluate_teukolsky_radial_line_rhs(
      grid, parameters, 0.9, consistent, angular, forcing,
      teuk::ReductionEvolution::FreeDamped, free_workspace, rhs_free);
  teuk::evaluate_teukolsky_radial_line_rhs(
      grid, parameters, 0.9, unconstrained, angular, forcing,
      teuk::ReductionEvolution::StageConstrained, constrained_workspace,
      rhs_constrained);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK_COMPLEX_NEAR(rhs_constrained[i].P, rhs_free[i].P, 3.0e-11);
    CHECK_COMPLEX_NEAR(rhs_constrained[i].Q, rhs_free[i].Q, 3.0e-11);
    CHECK_COMPLEX_NEAR(rhs_constrained[i].psi, rhs_free[i].psi, 3.0e-11);
  }
}

TEST_CASE("Teukolsky point kernels execute through the active Kokkos space") {
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.5;
  parameters.compactification_length = 2.0;
  parameters.azimuthal_mode = 2;
  Kokkos::View<teuk::Complex*> result("linear_point_result", 2);
  Kokkos::parallel_for(
      "linear_point_kernel", 1, KOKKOS_LAMBDA(const int) {
        const auto coefficients =
            teuk::teukolsky_coefficients(parameters, 0.25, 0.8);
        const teuk::TeukolskyState state{teuk::Complex(0.1, 0.2),
                                         teuk::Complex(-0.3, 0.4),
                                         teuk::Complex(0.5, -0.6)};
        result(0) = teuk::teukolsky_psi_rhs(coefficients, state);
        result(1) = teuk::teukolsky_p_rhs(
            coefficients, state, teuk::Complex(0.7, -0.8),
            teuk::Complex(-0.9, 1.0), teuk::Complex(1.1, -1.2));
      });
  const auto host_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  CHECK(Kokkos::isfinite(host_result(0).real()));
  CHECK(Kokkos::isfinite(host_result(0).imag()));
  CHECK(Kokkos::isfinite(host_result(1).real()));
  CHECK(Kokkos::isfinite(host_result(1).imag()));
}
