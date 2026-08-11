#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "teuk/config.hpp"
#include "teuk/coupled.hpp"
#include "teuk/diagnostics.hpp"
#include "teuk/rk4.hpp"

namespace {

void print_complex(const char* name, const teuk::Complex value) {
  std::cout << name << '=' << value.real() << ',' << value.imag() << '\n';
}

int run_point_pipeline(const teuk::Parameters& input) {
  teuk::PointPipelineParameters parameters;
  parameters.background =
      {input.mass, input.spin, input.compactification_scale};
  const double r_plus = teuk::outer_horizon_radius(input.mass, input.spin);
  const double horizon_R = input.compactification_scale *
                           input.compactification_scale / r_plus;
  parameters.radius = 0.5 * horizon_R;
  parameters.theta = 0.91;
  parameters.ell = 2;
  parameters.reduction_damping = input.reduction_damping;

  std::vector<teuk::PointPipelineState> state{
      teuk::make_point_pipeline_seed(1.0)};
  teuk::RK4Workspace<teuk::PointPipelineState> workspace(1);
  const auto rhs = [&](double,
                       const std::vector<teuk::PointPipelineState>& stage,
                       std::vector<teuk::PointPipelineState>& derivative) {
    derivative[0] = teuk::evaluate_point_pipeline_rhs(stage[0], parameters);
  };
  const double dt = input.final_time / static_cast<double>(input.steps);
  double time = 0.0;
  for (int step = 0; step < input.steps; ++step) {
    teuk::classical_rk4_step(state, time, dt, rhs, workspace);
    time += dt;
  }

  teuk::PointPipelineDiagnostics diagnostics;
  (void)teuk::evaluate_point_pipeline_rhs(state[0], parameters, &diagnostics);
  std::cout << "pipeline=point-common-stage-oracle\n"
            << "backend=" << Kokkos::DefaultExecutionSpace::name() << '\n'
            << "spin=" << input.spin << '\n'
            << "time=" << time << '\n'
            << "kappa=" << teuk::surface_gravity(input.mass, input.spin)
            << '\n'
            << "kappa_time="
            << teuk::surface_gravity(input.mass, input.spin) * time << '\n';
  print_complex("psi4_first", state[0].first.psi);
  print_complex("reconstruction_U", state[0].reconstruction.U);
  print_complex("quadratic_source", diagnostics.source_over_r3);
  print_complex("psi4_second", state[0].second.psi);
  std::cout << "limitation=single-point m=0 staging oracle; not a spatial science run\n";
  return 0;
}

void print_help() {
  std::cout
      << "Usage:\n"
      << "  teuk_solver backend\n"
      << "  teuk_solver point-pipeline [key=value ...]\n\n"
      << "Keys: mass spin L final_time steps gamma_q\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    const std::string_view command = argc > 1 ? argv[1] : "backend";
    if (command == "backend") {
      std::cout << "teuk-kokkos 0.1.0\n"
                << "Kokkos execution space: "
                << Kokkos::DefaultExecutionSpace::name() << '\n';
    } else if (command == "point-pipeline") {
      teuk::Parameters parameters;
      for (int i = 2; i < argc; ++i) {
        teuk::apply_key_value(parameters, argv[i]);
      }
      teuk::validate(parameters);
      status = run_point_pipeline(parameters);
    } else if (command == "help" || command == "--help" || command == "-h") {
      print_help();
    } else {
      std::cerr << "unknown command: " << command << '\n';
      print_help();
      status = 2;
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    status = 2;
  }
  Kokkos::finalize();
  return status;
}
