#pragma once

#include <string>
#include <vector>

#include "teuk/spatial_pipeline.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr int runtime_config_schema_version = 1;
inline constexpr const char* solver_executable_version = "0.2.0";

enum class InitialDataType { Gaussian, Checkpoint };

struct InitialMode {
  int ell = 2;
  int m = 2;
  Complex amplitude = Complex(1.0e-3, 0.0);
};

struct KerrRunParameters {
  double mass = 1.0;
  double spin = 0.0;
  double compactification_length = 1.0;
};

struct GridRunParameters {
  int radial_points = 65;
  int theta_points = 7;
  int ell_max_first = 4;
  int ell_max_second = 4;
  std::vector<int> first_order_modes{-2, 2};
  std::vector<int> second_order_modes{-2, 2};
};

struct TimeRunParameters {
  double final_time = 2.0e-3;
  int steps = 20;
  double cfl = 0.1;
};

struct MethodRunParameters {
  double reduction_damping = 0.1;
  double dissipation = 0.01;
  ReductionEvolution reduction = ReductionEvolution::FreeDamped;
};

struct InitialDataParameters {
  InitialDataType type = InitialDataType::Gaussian;
  std::vector<InitialMode> modes{{2, 2, Complex(1.0e-3, 0.0)}};
  bool add_sharp_partner = false;
  double center_fraction = 0.45;
  double width_fraction = 0.12;
  std::string time_derivative = "zero";
  std::string checkpoint_directory;
};

struct SecondOrderRunParameters {
  bool enabled = false;
  SecondOrderSourceMode source_mode =
      SecondOrderSourceMode::ConstraintAware;
  double source_start_time = 0.0;
  double normalized_constraint_tolerance = 1.0e-8;
  int required_consecutive_passes = 1;
  bool allow_truncated_daughter_modes = false;
};

struct OutputRunParameters {
  std::string directory = "teuk-output";
  int diagnostic_interval = 5;
  int checkpoint_interval = 0;
};

struct RunParameters {
  int config_version = runtime_config_schema_version;
  KerrRunParameters background;
  GridRunParameters grid;
  TimeRunParameters time;
  MethodRunParameters method;
  InitialDataParameters initial_data;
  SecondOrderRunParameters second_order;
  OutputRunParameters output;
};

}  // namespace teuk
