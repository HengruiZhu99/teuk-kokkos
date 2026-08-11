#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>

#include "teuk/diagnostics.hpp"
#include "teuk/horizon_diagnostics.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/spatial_pipeline.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const teuk::ExecutionSpace execution;
    const teuk::ModeRegistry registry({-4, -2, 0, 2, 4});
    const teuk::KerrParameters background{1.0, 0.999, 1.0};
    const double horizon = background.compactification_length *
                           background.compactification_length /
                           teuk::outer_horizon_radius(background.mass,
                                                      background.spin);
    const teuk::UniformRadialGrid radial_grid(129, 0.0, horizon);
    constexpr int theta_nodes = 7;
    teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 4,
                                   theta_nodes, background, 0.15, 0.0125);
    teuk::PipelineGaussianPulse pulse;
    pulse.center = 0.38 * horizon;
    pulse.width = 0.1 * horizon;
    pulse.modes = {{2, 2, teuk::Complex(5.0e-4, 1.0e-4)},
                   {2, -2, teuk::Complex(5.0e-4, -1.0e-4)}};
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, 4, background, pulse);

    constexpr double time_step = 5.0e-5;
    constexpr int steps = 40;
    double time = 0.0;
    for (int step = 0; step < steps; ++step) {
      pipeline.step(execution, time, time_step);
      time += time_step;
    }
    pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                  pipeline.storage().rhs(), time);
    teuk::PipelineDiagnostics diagnostics(registry.size(), radial_grid,
                                          theta_nodes);
    const auto report = diagnostics.sample_pipeline(execution, pipeline);
    teuk::HorizonTransverseDiagnostics horizon_output(
        radial_grid, registry.size(), theta_nodes);
    horizon_output.evaluate(
        execution, pipeline.storage().state(),
        static_cast<std::size_t>(teuk::PipelineField::FirstPsi));
    execution.fence("finish near-extremal horizon diagnostic");
    const auto horizon_values = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, horizon_output.values());
    std::array<double, teuk::horizon_derivative_count> maximum{};
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t derivative = 0;
           derivative < teuk::horizon_derivative_count; ++derivative) {
        for (int theta = 0; theta < theta_nodes; ++theta) {
          maximum[derivative] = std::max(
              maximum[derivative],
              static_cast<double>(Kokkos::abs(horizon_values(
                  mode, derivative, static_cast<std::size_t>(theta)))));
        }
      }
    }
    const double kappa = teuk::surface_gravity(background.mass,
                                               background.spin);
    std::cout << "backend=" << teuk::ExecutionSpace::name() << '\n'
              << "spin=" << background.spin << '\n'
              << "kappa=" << kappa << '\n'
              << "time=" << time << '\n'
              << "kappa_time=" << kappa * time << '\n'
              << "source_rms=" << report.source_over_r3.rms << '\n'
              << "source_active=" << report.second_order_source_active << '\n'
              << "independent_constraint_max="
              << report.independent_reconstruction_constraint_maximum << '\n'
              << "first_constraint_rms="
              << report.first_reduction_constraint.rms << '\n';
    for (std::size_t derivative = 0;
         derivative < teuk::horizon_derivative_count; ++derivative) {
      std::cout << "first_horizon_d" << derivative << '='
                << maximum[derivative] << '\n';
    }
    std::cout << "limitation=short diagnostic run; not an Aretakis claim\n";
  }
  Kokkos::finalize();
  return 0;
}
