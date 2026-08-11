#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <vector>

#include "teuk/pipeline_checkpoint.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/run_parameters.hpp"

namespace teuk {

inline ModeRegistry make_run_mode_registry(const RunParameters& parameters) {
  std::vector<int> stored = parameters.grid.first_order_modes;
  stored.insert(stored.end(), parameters.grid.second_order_modes.begin(),
                parameters.grid.second_order_modes.end());
  std::sort(stored.begin(), stored.end());
  stored.erase(std::unique(stored.begin(), stored.end()), stored.end());
  return ModeRegistry(stored, parameters.grid.first_order_modes,
                      parameters.grid.second_order_modes);
}

inline SecondOrderSourcePolicy make_source_policy(
    const RunParameters& parameters) {
  if (!parameters.second_order.enabled) {
    return SecondOrderSourcePolicy::disabled();
  }
  return {parameters.second_order.source_mode,
          parameters.second_order.source_start_time,
          parameters.second_order.normalized_constraint_tolerance,
          parameters.second_order.required_consecutive_passes};
}

inline PipelineCheckpointConfiguration make_checkpoint_configuration(
    const RunParameters& parameters, const double time_step) {
  return {{parameters.background.mass, parameters.background.spin,
           parameters.background.compactification_length},
          parameters.grid.ell_max_first,
          parameters.grid.ell_max_second,
          parameters.grid.theta_points,
          parameters.method.reduction_damping,
          parameters.method.dissipation,
          parameters.method.reduction,
          time_step,
          make_source_policy(parameters)};
}

struct InitializedPipelineState {
  double time = 0.0;
  std::uint64_t completed_steps = 0;
};

inline std::vector<GaussianPulseMode> expanded_gaussian_modes(
    const InitialDataParameters& parameters) {
  std::vector<GaussianPulseMode> modes;
  modes.reserve(parameters.modes.size() *
                (parameters.add_sharp_partner ? 2 : 1));
  for (const auto& mode : parameters.modes) {
    modes.push_back({mode.ell, mode.m, mode.amplitude});
    if (parameters.add_sharp_partner && mode.m != 0) {
      modes.push_back({mode.ell, -mode.m, Kokkos::conj(mode.amplitude)});
    }
  }
  std::sort(modes.begin(), modes.end(), [](const auto& left, const auto& right) {
    if (left.m != right.m) return left.m < right.m;
    return left.ell < right.ell;
  });
  if (std::adjacent_find(
          modes.begin(), modes.end(), [](const auto& left, const auto& right) {
            return left.m == right.m && left.ell == right.ell;
          }) != modes.end()) {
    throw std::invalid_argument(
        "expanded Gaussian modes contain duplicate (ell,m) seeds");
  }
  return modes;
}

inline InitializedPipelineState initialize_pipeline_state(
    const ExecutionSpace& execution, SpatialPipeline& pipeline,
    const ModeRegistry& registry, const RunParameters& parameters,
    const PipelineCheckpointConfiguration& checkpoint_configuration,
    const double horizon_radius) {
  const KerrParameters background{parameters.background.mass,
                                  parameters.background.spin,
                                  parameters.background.compactification_length};
  switch (parameters.initial_data.type) {
    case InitialDataType::Gaussian: {
      PipelineGaussianPulse pulse;
      pulse.center = parameters.initial_data.center_fraction * horizon_radius;
      pulse.width = parameters.initial_data.width_fraction * horizon_radius;
      pulse.modes = expanded_gaussian_modes(parameters.initial_data);
      initialize_compactified_gaussian_pulse(
          execution, pipeline, registry, parameters.grid.ell_max_first,
          background, pulse);
      return {};
    }
    case InitialDataType::Checkpoint: {
      const auto metadata = load_pipeline_checkpoint(
          execution, parameters.initial_data.checkpoint_directory, pipeline,
          registry, checkpoint_configuration);
      return {metadata.progress.time, metadata.progress.step};
    }
  }
  throw std::invalid_argument("unsupported initial-data factory type");
}

}  // namespace teuk
