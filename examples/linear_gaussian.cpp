#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <vector>

#include "teuk/diagnostics.hpp"
#include "teuk/grid.hpp"
#include "teuk/rk4.hpp"
#include "teuk/teukolsky.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr std::size_t point_count = 129;
    constexpr double mass = 1.0;
    constexpr double spin = 0.7;
    constexpr double length = 1.4;
    const double horizon_R =
        length * length / teuk::outer_horizon_radius(mass, spin);
    const teuk::UniformRadialGrid grid(point_count, horizon_R);
    teuk::TeukolskyParameters parameters;
    parameters.mass = mass;
    parameters.spin = spin;
    parameters.compactification_length = length;
    parameters.spin_weight = -2;
    parameters.azimuthal_mode = 2;
    parameters.reduction_damping = 0.2;
    constexpr double theta = 0.9;
    constexpr double angular_eigenvalue = -2.0;

    std::vector<teuk::TeukolskyState> state(point_count);
    std::vector<teuk::Complex> psi(point_count), derivative(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
      const double R = grid.coordinate(i);
      const double centered = (R - 0.45 * horizon_R) / (0.12 * horizon_R);
      psi[i] = std::exp(-centered * centered);
    }
    teuk::fourth_order_radial_derivative(grid, psi, derivative);
    for (std::size_t i = 0; i < point_count; ++i) {
      const auto coefficients =
          teuk::teukolsky_coefficients(parameters, grid.coordinate(i), theta);
      state[i].psi = psi[i];
      state[i].Q = derivative[i];
      state[i].P = -2.0 * coefficients.radial_advection * state[i].Q +
                   coefficients.definition * state[i].psi;
    }

    std::vector<teuk::Complex> angular(point_count), forcing(point_count, 0.0);
    teuk::TeukolskyRadialWorkspace radial_workspace(point_count);
    teuk::RK4Workspace<teuk::TeukolskyState> rk_workspace(point_count);
    const auto rhs = [&](double, const std::vector<teuk::TeukolskyState>& in,
                         std::vector<teuk::TeukolskyState>& out) {
      for (std::size_t i = 0; i < point_count; ++i) {
        angular[i] = angular_eigenvalue * in[i].psi;
      }
      teuk::evaluate_teukolsky_radial_line_rhs(
          grid, parameters, theta, in, angular, forcing,
          teuk::ReductionEvolution::FreeDamped, radial_workspace, out);
    };
    constexpr double dt = 2.0e-5;
    constexpr int steps = 50;
    double time = 0.0;
    for (int step = 0; step < steps; ++step) {
      teuk::classical_rk4_step(state, time, dt, rhs, rk_workspace);
      time += dt;
    }
    std::cout << "backend=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "time=" << time << '\n'
              << "constraint_rms="
              << teuk::reduction_constraint_rms(grid, state, psi, derivative)
              << '\n'
              << "note=reference FD pulse without production SAT boundaries\n";
  }
  Kokkos::finalize();
  return 0;
}
