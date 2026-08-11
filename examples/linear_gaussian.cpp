#include <Kokkos_Core.hpp>

#include <iostream>

#include "teuk/diagnostics.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/spatial_pipeline.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const teuk::ExecutionSpace execution;
    const teuk::ModeRegistry registry({-2, 2});
    const teuk::KerrParameters background{1.0, 0.7, 1.4};
    const double horizon = background.compactification_length *
                           background.compactification_length /
                           teuk::outer_horizon_radius(background.mass,
                                                      background.spin);
    const teuk::UniformRadialGrid radial_grid(65, 0.0, horizon);
    teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 3, 5,
                                   background, 0.2, 0.01);
    teuk::PipelineGaussianPulse pulse;
    pulse.center = 0.45 * horizon;
    pulse.width = 0.12 * horizon;
    pulse.modes = {{2, 2, teuk::Complex(1.0e-3, 2.0e-4)},
                   {2, -2, teuk::Complex(1.0e-3, -2.0e-4)}};
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, 3, background, pulse);

    constexpr double time_step = 1.0e-4;
    constexpr int steps = 20;
    double time = 0.0;
    for (int step = 0; step < steps; ++step) {
      pipeline.step(execution, time, time_step);
      time += time_step;
    }
    pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                  pipeline.storage().rhs(), time);
    teuk::PipelineDiagnostics diagnostics(registry.size(), radial_grid, 5);
    const auto report = diagnostics.sample_pipeline(execution, pipeline);
    std::cout << "backend=" << teuk::ExecutionSpace::name() << '\n'
              << "time=" << time << '\n'
              << "first_psi4_rms="
              << report.fields[static_cast<std::size_t>(
                                    teuk::PipelineField::FirstPsi)]
                     .state.rms
              << '\n'
              << "first_constraint_rms="
              << report.first_reduction_constraint.rms << '\n'
              << "source_active=" << report.second_order_source_active << '\n'
              << "independent_constraint_max="
              << report.independent_reconstruction_constraint_maximum << '\n'
              << "second_psi4_rms="
              << report.fields[static_cast<std::size_t>(
                                    teuk::PipelineField::SecondPsi)]
                     .state.rms
              << '\n'
              << "note=constraint-aware startup suppresses unqualified "
                 "second-order forcing; this is the linear full-grid path\n";
  }
  Kokkos::finalize();
  return 0;
}
