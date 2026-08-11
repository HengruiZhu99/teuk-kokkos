#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

#include "teuk/boundary.hpp"
#include "teuk/config.hpp"
#include "teuk/coupled.hpp"
#include "teuk/diagnostics.hpp"
#include "teuk/horizon_diagnostics.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_checkpoint.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/pipeline_reconstruction_diagnostics.hpp"
#include "teuk/rk4.hpp"
#include "teuk/spatial_pipeline.hpp"

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

double maximum_coordinate_speed(const teuk::Parameters& input,
                                const teuk::UniformRadialGrid& grid) {
  teuk::TeukolskyParameters parameters;
  parameters.mass = input.mass;
  parameters.spin = input.spin;
  parameters.compactification_length = input.compactification_scale;
  parameters.spin_weight = -2;
  const auto angular = teuk::angular::gauss_legendre(input.theta_points);
  double maximum = 0.0;
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    for (int theta = 0; theta < input.theta_points; ++theta) {
      const auto characteristics = teuk::radial_principal_characteristics(
          parameters, grid.coordinate(radial), angular.theta(theta));
      maximum = std::max(
          maximum,
          std::max(std::abs(characteristics.coordinate_velocity_minus),
                   std::abs(characteristics.coordinate_velocity_plus)));
    }
  }
  return maximum;
}

std::filesystem::path checkpoint_path(const std::filesystem::path& output,
                                      const std::uint64_t step) {
  std::ostringstream name;
  name << "checkpoint_" << std::setw(8) << std::setfill('0') << step;
  return output / name.str();
}

int run_spatial_pipeline(const teuk::Parameters& input) {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry(input.modes);
  const teuk::KerrParameters background{
      input.mass, input.spin, input.compactification_scale};
  const double horizon =
      input.compactification_scale * input.compactification_scale /
      teuk::outer_horizon_radius(input.mass, input.spin);
  const teuk::UniformRadialGrid radial_grid(
      static_cast<std::size_t>(input.radial_points), 0.0, horizon);
  teuk::SpatialPipeline pipeline(
      execution, registry, radial_grid, input.ell_max, input.theta_points,
      background, input.reduction_damping, input.dissipation);

  const double time_step =
      input.final_time / static_cast<double>(input.steps);
  const double maximum_speed = maximum_coordinate_speed(input, radial_grid);
  const double cfl_limit = input.cfl * radial_grid.spacing() / maximum_speed;
  if (time_step > cfl_limit) {
    throw std::invalid_argument(
        "steps are too small for the requested spatial-pipeline CFL limit");
  }
  const teuk::PipelineCheckpointConfiguration checkpoint_configuration{
      background,
      input.ell_max,
      input.theta_points,
      input.reduction_damping,
      input.dissipation,
      teuk::ReductionEvolution::FreeDamped,
      time_step};
  double time = 0.0;
  std::uint64_t completed_steps = 0;
  if (input.restart_directory.empty()) {
    teuk::PipelineGaussianPulse pulse;
    pulse.center = input.pulse_center_fraction * horizon;
    pulse.width = input.pulse_width_fraction * horizon;
    const teuk::Complex seed_amplitude(input.pulse_amplitude,
                                       0.25 * input.pulse_amplitude);
    pulse.modes.push_back(
        {input.seed_ell, input.seed_mode, seed_amplitude});
    if (input.seed_mode != 0) {
      pulse.modes.push_back(
          {input.seed_ell, -input.seed_mode, Kokkos::conj(seed_amplitude)});
    }
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, input.ell_max, background, pulse);
  } else {
    const auto metadata = teuk::load_pipeline_checkpoint(
        execution, input.restart_directory, pipeline, registry,
        checkpoint_configuration);
    time = metadata.progress.time;
    completed_steps = metadata.progress.step;
  }

  teuk::PipelineDiagnostics diagnostics(
      registry.size(), radial_grid,
      static_cast<std::size_t>(input.theta_points));
  teuk::HorizonTransverseDiagnostics horizon_diagnostics(
      radial_grid, registry.size(),
      static_cast<std::size_t>(input.theta_points));
  teuk::PipelineReconstructionDiagnostics reconstruction_diagnostics(
      registry.size(), radial_grid.size(),
      static_cast<std::size_t>(input.theta_points));

  std::ofstream diagnostic_file;
  std::ofstream source_pair_file;
  std::filesystem::path output_directory;
  if (!input.output_directory.empty()) {
    output_directory = input.output_directory;
    std::filesystem::create_directories(output_directory);
    diagnostic_file.open(output_directory / "diagnostics.csv");
    if (!diagnostic_file) {
      throw std::runtime_error("cannot open spatial diagnostic output");
    }
    source_pair_file.open(output_directory / "source_pairs.csv");
    if (!source_pair_file) {
      throw std::runtime_error("cannot open spatial pair-source output");
    }
    source_pair_file
        << "step,time,pair,m1,m2,target,D_rms,D_max,T_rms,T_max\n";
  }

  constexpr std::size_t first_psi =
      static_cast<std::size_t>(teuk::PipelineField::FirstPsi);
  constexpr std::size_t second_psi =
      static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
  const auto horizon_maxima = [&](const std::size_t field) {
    horizon_diagnostics.evaluate(execution, pipeline.storage().state(), field);
    execution.fence("sample horizon transverse derivatives");
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, horizon_diagnostics.values());
    std::array<double, teuk::horizon_derivative_count> maximum{};
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t derivative = 0;
           derivative < teuk::horizon_derivative_count; ++derivative) {
        for (int theta = 0; theta < input.theta_points; ++theta) {
          maximum[derivative] =
              std::max(maximum[derivative],
                       static_cast<double>(Kokkos::abs(host(
                           mode, derivative, static_cast<std::size_t>(theta)))));
        }
      }
    }
    return maximum;
  };

  const char* header =
      "step,time,kappa_time,first_psi_rms,second_psi_rms,"
      "first_constraint_rms,second_constraint_rms,source_rms,forcing_rms,"
      "reconstruction_residual_rms,"
      "first_horizon_d0,first_horizon_d1,first_horizon_d2,"
      "first_horizon_d3,first_horizon_d4,second_horizon_d0,"
      "second_horizon_d1,second_horizon_d2,second_horizon_d3,"
      "second_horizon_d4";
  std::cout << header << '\n';
  if (diagnostic_file) diagnostic_file << header << '\n';
  const double kappa = teuk::surface_gravity(input.mass, input.spin);
  const auto sample = [&](const std::uint64_t step, const double time) {
    pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                          pipeline.storage().rhs());
    const auto report = diagnostics.sample_pipeline(execution, pipeline);
    const auto reconstruction =
        reconstruction_diagnostics.sample(execution, pipeline);
    const auto first_horizon = horizon_maxima(first_psi);
    const auto second_horizon = horizon_maxima(second_psi);
    std::ostringstream line;
    line << std::setprecision(17) << step << ',' << time << ','
         << kappa * time << ',' << report.fields[first_psi].state.rms << ','
         << report.fields[second_psi].state.rms << ','
         << report.first_reduction_constraint.rms << ','
         << report.second_reduction_constraint.rms << ','
         << report.source_over_r3.rms << ',' << report.forcing.rms << ','
         << reconstruction.combined.rms;
    for (const double value : first_horizon) line << ',' << value;
    for (const double value : second_horizon) line << ',' << value;
    std::cout << line.str() << '\n';
    if (diagnostic_file) diagnostic_file << line.str() << '\n';
    if (source_pair_file) {
      const auto per_pair = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, pipeline.per_pair_source());
      const double inverse_points =
          1.0 / static_cast<double>(radial_grid.size() *
                                    static_cast<std::size_t>(
                                        input.theta_points));
      for (std::size_t pair_index = 0;
           pair_index < registry.ordered_pairs().size(); ++pair_index) {
        std::array<double, 2> squared_norm{};
        std::array<double, 2> maximum{};
        for (std::size_t component = 0; component < 2; ++component) {
          for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
            for (int theta = 0; theta < input.theta_points; ++theta) {
              const double magnitude = Kokkos::abs(per_pair(
                  pair_index, component, radial,
                  static_cast<std::size_t>(theta)));
              squared_norm[component] += magnitude * magnitude;
              maximum[component] = std::max(maximum[component], magnitude);
            }
          }
        }
        const auto& pair = registry.ordered_pairs()[pair_index];
        source_pair_file << std::setprecision(17) << step << ',' << time << ','
                         << pair_index << ',' << pair.m1 << ',' << pair.m2
                         << ',' << pair.target << ','
                         << std::sqrt(squared_norm[0] * inverse_points) << ','
                         << maximum[0] << ','
                         << std::sqrt(squared_norm[1] * inverse_points) << ','
                         << maximum[1] << '\n';
      }
    }
  };

  const auto total_start = std::chrono::steady_clock::now();
  sample(completed_steps, time);
  double evolution_wall_seconds = 0.0;
  auto evolution_start = std::chrono::steady_clock::now();
  for (int local_step = 1; local_step <= input.steps; ++local_step) {
    const std::uint64_t step =
        completed_steps + static_cast<std::uint64_t>(local_step);
    pipeline.step(execution, time, time_step);
    time += time_step;
    const bool diagnostic_step =
        local_step % input.diagnostic_interval == 0 ||
        local_step == input.steps;
    const bool checkpoint_step =
        input.checkpoint_interval > 0 &&
        (local_step % input.checkpoint_interval == 0 ||
         local_step == input.steps);
    if (diagnostic_step || checkpoint_step) {
      execution.fence("time full spatial pipeline evolution segment");
      evolution_wall_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        evolution_start)
              .count();
    }
    if (diagnostic_step) sample(step, time);
    if (checkpoint_step) {
      if (output_directory.empty()) {
        throw std::invalid_argument(
            "checkpoint_every requires a nonempty output directory");
      }
      teuk::write_pipeline_checkpoint(
          execution, checkpoint_path(output_directory, step), pipeline,
          registry, checkpoint_configuration,
          {time, step});
    }
    if (diagnostic_step || checkpoint_step) {
      evolution_start = std::chrono::steady_clock::now();
    }
  }
  execution.fence("finish full spatial pipeline run");
  const double total_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    total_start)
          .count();
  const double point_steps =
      static_cast<double>(input.steps) * registry.size() * radial_grid.size() *
      static_cast<double>(input.theta_points);
  teuk::SpatialPipelineTiming profile;
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs(), &profile);
  const double profile_total = profile.total_seconds();
  const auto percentage = [&](const double seconds) {
    return 100.0 * seconds / profile_total;
  };
  std::cout << "backend=" << teuk::ExecutionSpace::name() << '\n'
            << "boundary_policy=zero-SAT; no incoming propagating modes\n"
            << "evolution_wall_seconds=" << evolution_wall_seconds << '\n'
            << "total_wall_seconds=" << total_wall_seconds << '\n'
            << "grid_point_steps_per_second="
            << point_steps / evolution_wall_seconds << '\n'
            << "profile_first_linear_percent="
            << percentage(profile.first_linear_seconds) << '\n'
            << "profile_reconstruction_percent="
            << percentage(profile.reconstruction_seconds) << '\n'
            << "profile_tangent_percent="
            << percentage(profile.tangent_seconds) << '\n'
            << "profile_source_percent="
            << percentage(profile.source_seconds) << '\n'
            << "profile_second_linear_percent="
            << percentage(profile.second_linear_seconds) << '\n';
  return 0;
}

void print_help() {
  std::cout
      << "Usage:\n"
      << "  teuk_solver backend\n"
      << "  teuk_solver point-pipeline [key=value ...]\n"
      << "  teuk_solver spatial-pipeline [key=value ...]\n\n"
      << "Keys: mass spin L nr ntheta ellmax modes seed_ell seed_m cfl\n"
      << "      final_time steps gamma_q dissipation amplitude pulse_center\n"
      << "      pulse_width diagnostic_every checkpoint_every output restart\n";
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
    } else if (command == "spatial-pipeline") {
      teuk::Parameters parameters;
      for (int i = 2; i < argc; ++i) {
        teuk::apply_key_value(parameters, argv[i]);
      }
      teuk::validate(parameters);
      status = run_spatial_pipeline(parameters);
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
