#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>

#include "teuk/diagnostics.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/spatial_pipeline.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const teuk::ExecutionSpace execution;
    const teuk::ModeRegistry registry({-4, -2, 0, 2, 4});
    const teuk::KerrParameters background{1.0, 0.8, 1.3};
    const double horizon = background.compactification_length *
                           background.compactification_length /
                           teuk::outer_horizon_radius(background.mass,
                                                      background.spin);
    const teuk::UniformRadialGrid radial_grid(65, 0.0, horizon);
    constexpr int theta_nodes = 7;
    teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 4,
                                   theta_nodes, background, 0.15, 0.01);
    teuk::PipelineGaussianPulse pulse;
    pulse.center = 0.42 * horizon;
    pulse.width = 0.11 * horizon;
    pulse.modes = {{2, 2, teuk::Complex(8.0e-4, 2.0e-4)},
                   {2, -2, teuk::Complex(8.0e-4, -2.0e-4)}};
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, 4, background, pulse);

    constexpr double time_step = 1.0e-4;
    constexpr int steps = 20;
    double time = 0.0;
    for (int step = 0; step < steps; ++step) {
      pipeline.step(execution, time, time_step);
      time += time_step;
    }
    execution.fence("finish daughter-mode example");
    const auto state = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, pipeline.storage().state());
    const std::size_t second_psi =
        static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
    std::cout << "backend=" << teuk::ExecutionSpace::name() << '\n'
              << "time=" << time << '\n'
              << "m,second_psi4_rms\n";
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      double sum = 0.0;
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (int theta = 0; theta < theta_nodes; ++theta) {
          const double magnitude = Kokkos::abs(
              state(mode, second_psi, radial,
                    static_cast<std::size_t>(theta)));
          sum += magnitude * magnitude;
        }
      }
      const double count =
          static_cast<double>(radial_grid.size() * theta_nodes);
      std::cout << registry.modes()[mode] << ',' << std::sqrt(sum / count)
                << '\n';
    }
    std::cout << "expected_daughters=-4,0,4\n";
  }
  Kokkos::finalize();
  return 0;
}
