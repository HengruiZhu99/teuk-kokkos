#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

#include "teuk/coupled.hpp"
#include "teuk/diagnostics.hpp"
#include "teuk/rk4.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    teuk::PointPipelineParameters parameters;
    parameters.background = {1.0, 0.999, 1.0};
    parameters.radius = 0.5;
    parameters.theta = 0.91;
    parameters.ell = 2;
    std::vector<teuk::PointPipelineState> state{
        teuk::make_point_pipeline_seed(1.0e-3)};
    teuk::RK4Workspace<teuk::PointPipelineState> workspace(1);
    const auto rhs = [&](double,
                         const std::vector<teuk::PointPipelineState>& in,
                         std::vector<teuk::PointPipelineState>& out) {
      out[0] = teuk::evaluate_point_pipeline_rhs(in[0], parameters);
    };
    constexpr double dt = 1.0e-3;
    constexpr int steps = 100;
    double time = 0.0;
    for (int step = 0; step < steps; ++step) {
      teuk::classical_rk4_step(state, time, dt, rhs, workspace);
      time += dt;
    }
    teuk::PointPipelineDiagnostics diagnostics;
    (void)teuk::evaluate_point_pipeline_rhs(state[0], parameters, &diagnostics);
    const double kappa = teuk::surface_gravity(1.0, 0.999);
    std::cout << "spin=0.999\nkappa=" << kappa
              << "\ntime=" << time << "\nkappa_time=" << kappa * time
              << "\nabs_source=" << Kokkos::abs(diagnostics.source_over_r3)
              << "\nabs_psi4_second=" << Kokkos::abs(state[0].second.psi)
              << "\nlimitation=point staging diagnostic, not an Aretakis run\n";
  }
  Kokkos::finalize();
  return 0;
}
