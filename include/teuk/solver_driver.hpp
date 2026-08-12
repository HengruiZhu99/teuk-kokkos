#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "teuk/boundary.hpp"
#include "teuk/background.hpp"
#include "teuk/config.hpp"
#include "teuk/diagnostics.hpp"
#include "teuk/horizon_diagnostics.hpp"
#include "teuk/initial_data_factory.hpp"
#include "teuk/pipeline_diagnostics.hpp"
#include "teuk/pipeline_independent_reconstruction_diagnostics.hpp"
#include "teuk/pipeline_reconstruction_diagnostics.hpp"
#include "teuk/waveform_output.hpp"

#ifndef TEUK_GIT_COMMIT
#define TEUK_GIT_COMMIT "unknown"
#endif

namespace teuk {

inline double maximum_coordinate_speed(const RunParameters& input,
                                       const UniformRadialGrid& grid) {
  TeukolskyParameters parameters;
  parameters.mass = input.background.mass;
  parameters.spin = input.background.spin;
  parameters.compactification_length =
      input.background.compactification_length;
  parameters.spin_weight = -2;
  const auto angular = angular::gauss_legendre(input.grid.theta_points);
  double maximum = 0.0;
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    for (int theta = 0; theta < input.grid.theta_points; ++theta) {
      const auto characteristics = radial_principal_characteristics(
          parameters, grid.coordinate(radial), angular.theta(theta));
      maximum = std::max(
          maximum,
          std::max(std::abs(characteristics.coordinate_velocity_minus),
                   std::abs(characteristics.coordinate_velocity_plus)));
    }
  }
  return maximum;
}

inline std::filesystem::path pipeline_checkpoint_path(
    const std::filesystem::path& output, const std::uint64_t step) {
  std::ostringstream name;
  name << "checkpoint_" << std::setw(8) << std::setfill('0') << step;
  return output / name.str();
}

inline void write_resolved_run_configuration(
    const RunParameters& parameters) {
  std::filesystem::create_directories(parameters.output.directory);
  std::ofstream output(std::filesystem::path(parameters.output.directory) /
                       "resolved_config.cfg");
  if (!output) throw std::runtime_error("cannot write resolved_config.cfg");
  output << "# teuk_solver executable_version = " << solver_executable_version
         << '\n'
         << "# git_commit = " << TEUK_GIT_COMMIT << '\n'
         << "# kokkos_backend = " << ExecutionSpace::name() << '\n'
         << "# kokkos_version = " << KOKKOS_VERSION_MAJOR << '.'
         << KOKKOS_VERSION_MINOR << '.' << KOKKOS_VERSION_PATCH << '\n'
         << "# build_precision = complex_binary64\n"
         << "# config_schema_version = " << runtime_config_schema_version
         << '\n'
         << resolved_configuration_text(parameters);
  output.flush();
  if (!output) throw std::runtime_error("failed writing resolved_config.cfg");
}

inline int run_solver(const RunParameters& input) {
  validate_run_parameters(input);
  write_resolved_run_configuration(input);
  if (input.second_order.enabled) {
    const auto completeness = validate_quadratic_daughter_modes(
        input.grid.first_order_modes, input.grid.second_order_modes,
        input.second_order.allow_truncated_daughter_modes);
    if (!completeness.complete()) {
      std::cerr << "WARNING: second-order evolution truncates daughter modes:";
      for (const int mode : completeness.missing) std::cerr << ' ' << mode;
      std::cerr << "\nWARNING: this run is not mode-complete.\n";
    }
  }

  const ExecutionSpace execution;
  const ModeRegistry registry = make_run_mode_registry(input);
  const KerrParameters background{
      input.background.mass, input.background.spin,
      input.background.compactification_length};
  const double horizon =
      input.background.compactification_length *
      input.background.compactification_length /
      outer_horizon_radius(input.background.mass, input.background.spin);
  const UniformRadialGrid radial_grid(
      static_cast<std::size_t>(input.grid.radial_points), 0.0, horizon);
  const double time_step =
      input.time.final_time / static_cast<double>(input.time.steps);
  const double maximum_speed = maximum_coordinate_speed(input, radial_grid);
  const double cfl_limit =
      input.time.cfl * radial_grid.spacing() / maximum_speed;
  if (time_step > cfl_limit) {
    throw std::invalid_argument(
        "steps are too small for the requested radial CFL limit");
  }

  SpatialPipeline pipeline(
      execution, registry, radial_grid,
      {input.grid.ell_max_first, input.grid.ell_max_second},
      input.grid.theta_points, background, input.method.reduction_damping,
      input.method.dissipation, input.method.reduction, "full_pipeline",
      make_source_policy(input));
  const auto checkpoint_configuration =
      make_checkpoint_configuration(input, time_step);
  const auto initialized = initialize_pipeline_state(
      execution, pipeline, registry, input, checkpoint_configuration, horizon);
  double time = initialized.time;
  const std::uint64_t completed_steps = initialized.completed_steps;

  PipelineDiagnostics diagnostics(
      registry.size(), radial_grid,
      static_cast<std::size_t>(input.grid.theta_points));
  HorizonTransverseDiagnostics horizon_diagnostics(
      radial_grid, registry.size(),
      static_cast<std::size_t>(input.grid.theta_points));
  PipelineTransportEquationDiagnostics transport_diagnostics(
      registry.size(), radial_grid.size(),
      static_cast<std::size_t>(input.grid.theta_points));
  PipelineIndependentReconstructionDiagnostics independent_diagnostics(
      registry.size(), radial_grid.size(),
      static_cast<std::size_t>(input.grid.theta_points));

  const std::filesystem::path output_directory = input.output.directory;
  std::ofstream diagnostic_file(output_directory / "diagnostics.csv");
  std::ofstream source_pair_file(output_directory / "source_pairs.csv");
  std::ofstream waveform_file(output_directory / "waveforms.csv");
  if (!diagnostic_file || !source_pair_file || !waveform_file) {
    throw std::runtime_error("cannot open spatial diagnostic output");
  }
  source_pair_file
      << "step,time,pair,m1,m2,target,D_rms,D_max,T_rms,T_max\n";
  write_endpoint_waveform_header(waveform_file);
  EndpointWaveformSampler waveform_sampler(
      registry,
      {input.grid.ell_max_first, input.grid.ell_max_second},
      input.grid.theta_points, input.background.compactification_length,
      horizon);

  constexpr std::size_t first_psi =
      static_cast<std::size_t>(PipelineField::FirstPsi);
  constexpr std::size_t second_psi =
      static_cast<std::size_t>(PipelineField::SecondPsi);
  const auto horizon_maxima = [&](const std::size_t field) {
    horizon_diagnostics.evaluate(execution, pipeline.storage().state(), field);
    execution.fence("sample horizon transverse derivatives");
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, horizon_diagnostics.values());
    std::array<double, horizon_derivative_count> maximum{};
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t derivative = 0;
           derivative < horizon_derivative_count; ++derivative) {
        for (int theta = 0; theta < input.grid.theta_points; ++theta) {
          maximum[derivative] = std::max(
              maximum[derivative],
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
      "source_active,source_activation_time,source_constraint_abs_max,"
      "source_constraint_normalized_max,source_constraint_normalized_l2,"
      "transport_residual_rms,psi3_bianchi_rms,psi2_bianchi_rms,"
      "hll_reality_rms,first_horizon_d0,first_horizon_d1,first_horizon_d2,"
      "first_horizon_d3,first_horizon_d4,second_horizon_d0,"
      "second_horizon_d1,second_horizon_d2,second_horizon_d3,"
      "second_horizon_d4";
  std::cout << header << '\n';
  diagnostic_file << header << '\n';
  const double kappa =
      surface_gravity(input.background.mass, input.background.spin);
  const auto sample = [&](const std::uint64_t step, const double sample_time) {
    pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                  pipeline.storage().rhs(), sample_time);
    const auto report = diagnostics.sample_pipeline(execution, pipeline);
    const auto transport = transport_diagnostics.sample(execution, pipeline);
    const auto independent = independent_diagnostics.sample(execution, pipeline);
    const auto first_horizon = horizon_maxima(first_psi);
    const auto second_horizon = horizon_maxima(second_psi);
    const auto& normalized = report.normalized_source_constraints;
    const double normalized_maximum = std::max(
        {normalized.psi3_bianchi.normalized_maximum,
         normalized.psi2_bianchi.normalized_maximum,
         normalized.hll_reality.normalized_maximum});
    const double normalized_l2 = std::max(
        {normalized.psi3_bianchi.normalized_weighted,
         normalized.psi2_bianchi.normalized_weighted,
         normalized.hll_reality.normalized_weighted});
    std::ostringstream line;
    line << std::setprecision(17) << step << ',' << sample_time << ','
         << kappa * sample_time << ',' << report.fields[first_psi].state.rms
         << ',' << report.fields[second_psi].state.rms << ','
         << report.first_reduction_constraint.rms << ','
         << report.second_reduction_constraint.rms << ','
         << report.source_over_r3.rms << ',' << report.forcing.rms << ','
         << (report.second_order_source_active ? 1 : 0) << ','
         << report.source_activation.activation_time << ','
         << report.independent_reconstruction_constraint_maximum << ','
         << normalized_maximum << ',' << normalized_l2 << ','
         << transport.combined.rms << ',' << independent.psi3_bianchi.rms
         << ',' << independent.psi2_bianchi.rms << ','
         << independent.hll_reality.rms;
    for (const double value : first_horizon) line << ',' << value;
    for (const double value : second_horizon) line << ',' << value;
    std::cout << line.str() << '\n';
    diagnostic_file << line.str() << '\n';
    write_endpoint_waveform_records(
        waveform_file,
        waveform_sampler.sample(execution, pipeline.storage().state(), step,
                                sample_time));
    waveform_file.flush();
    if (!waveform_file) {
      throw std::runtime_error("failed writing endpoint waveform output");
    }

    const auto per_pair = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, pipeline.per_pair_source());
    const double inverse_points =
        1.0 / static_cast<double>(radial_grid.size() *
                                  static_cast<std::size_t>(
                                      input.grid.theta_points));
    for (std::size_t pair_index = 0;
         pair_index < registry.ordered_pairs().size(); ++pair_index) {
      std::array<double, 2> squared_norm{};
      std::array<double, 2> maximum{};
      for (std::size_t component = 0; component < 2; ++component) {
        for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
          for (int theta = 0; theta < input.grid.theta_points; ++theta) {
            const double magnitude = Kokkos::abs(per_pair(
                pair_index, component, radial,
                static_cast<std::size_t>(theta)));
            squared_norm[component] += magnitude * magnitude;
            maximum[component] = std::max(maximum[component], magnitude);
          }
        }
      }
      const auto& pair = registry.ordered_pairs()[pair_index];
      source_pair_file
          << std::setprecision(17) << step << ',' << sample_time << ','
          << pair_index << ',' << pair.m1 << ',' << pair.m2 << ','
          << pair.target << ','
          << std::sqrt(squared_norm[0] * inverse_points) << ','
          << maximum[0] << ','
          << std::sqrt(squared_norm[1] * inverse_points) << ','
          << maximum[1] << '\n';
    }
  };

  const auto total_start = std::chrono::steady_clock::now();
  sample(completed_steps, time);
  double evolution_wall_seconds = 0.0;
  auto evolution_start = std::chrono::steady_clock::now();
  for (int local_step = 1; local_step <= input.time.steps; ++local_step) {
    const std::uint64_t step =
        completed_steps + static_cast<std::uint64_t>(local_step);
    pipeline.step(execution, time, time_step);
    time += time_step;
    const bool diagnostic_step =
        local_step % input.output.diagnostic_interval == 0 ||
        local_step == input.time.steps;
    const bool checkpoint_step =
        input.output.checkpoint_interval > 0 &&
        (local_step % input.output.checkpoint_interval == 0 ||
         local_step == input.time.steps);
    if (diagnostic_step || checkpoint_step) {
      execution.fence("time full spatial pipeline evolution segment");
      evolution_wall_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        evolution_start)
              .count();
    }
    if (diagnostic_step) sample(step, time);
    if (checkpoint_step) {
      write_pipeline_checkpoint(
          execution, pipeline_checkpoint_path(output_directory, step), pipeline,
          registry, checkpoint_configuration, {time, step});
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
      static_cast<double>(input.time.steps) * registry.size() *
      radial_grid.size() * static_cast<double>(input.grid.theta_points);
  SpatialPipelineTiming profile;
  pipeline.evaluate_rhs_at_time(execution, pipeline.storage().state(),
                                pipeline.storage().rhs(), time, &profile);
  const double profile_total = profile.total_seconds();
  const auto percentage = [&](const double seconds) {
    return profile_total > 0.0 ? 100.0 * seconds / profile_total : 0.0;
  };
  std::cout << "backend=" << ExecutionSpace::name() << '\n'
            << "boundary_policy=zero-SAT; no incoming propagating modes\n"
            << "source_policy="
            << config_detail::source_mode_name(pipeline.source_policy().mode)
            << '\n'
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
            << "profile_source_percent=" << percentage(profile.source_seconds)
            << '\n'
            << "profile_second_linear_percent="
            << percentage(profile.second_linear_seconds) << '\n';
  return 0;
}

}  // namespace teuk
