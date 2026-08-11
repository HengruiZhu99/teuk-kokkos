#pragma once

#include <charconv>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace teuk {

struct Parameters {
  double mass = 1.0;
  double spin = 0.9;
  double compactification_scale = 1.0;
  int radial_points = 128;
  int theta_points = 16;
  int ell_max = 6;
  int seed_ell = 2;
  int seed_mode = 2;
  double cfl = 0.1;
  double final_time = 1.0;
  double reduction_damping = 0.1;
  double dissipation = 0.01;
  double pulse_amplitude = 1.0e-3;
  double pulse_center_fraction = 0.45;
  double pulse_width_fraction = 0.12;
  int steps = 100;
  int diagnostic_interval = 10;
  int checkpoint_interval = 0;
  std::vector<int> modes{-4, -2, 0, 2, 4};
  std::string output_directory;
};

inline double parse_double(std::string_view text, const char* key) {
  double value = 0.0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument(std::string("invalid value for ") + key);
  }
  return value;
}

inline int parse_int(std::string_view text, const char* key) {
  int value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument(std::string("invalid value for ") + key);
  }
  return value;
}

inline std::vector<int> parse_modes(std::string_view text) {
  std::vector<int> result;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string_view token = text.substr(
        begin, end == std::string_view::npos ? text.size() - begin
                                             : end - begin);
    if (token.empty()) throw std::invalid_argument("invalid value for modes");
    result.push_back(parse_int(token, "modes"));
    if (end == std::string_view::npos) break;
    begin = end + 1;
  }
  if (result.empty()) throw std::invalid_argument("modes must not be empty");
  return result;
}

inline void apply_key_value(Parameters& parameters, std::string_view argument) {
  const std::size_t delimiter = argument.find('=');
  if (delimiter == std::string_view::npos) {
    throw std::invalid_argument("expected key=value argument");
  }
  const std::string_view key = argument.substr(0, delimiter);
  const std::string_view value = argument.substr(delimiter + 1);

  if (key == "mass") parameters.mass = parse_double(value, "mass");
  else if (key == "spin") parameters.spin = parse_double(value, "spin");
  else if (key == "L") parameters.compactification_scale = parse_double(value, "L");
  else if (key == "nr") parameters.radial_points = parse_int(value, "nr");
  else if (key == "ntheta") parameters.theta_points = parse_int(value, "ntheta");
  else if (key == "ellmax") parameters.ell_max = parse_int(value, "ellmax");
  else if (key == "seed_ell") parameters.seed_ell = parse_int(value, "seed_ell");
  else if (key == "seed_m") parameters.seed_mode = parse_int(value, "seed_m");
  else if (key == "cfl") parameters.cfl = parse_double(value, "cfl");
  else if (key == "final_time") parameters.final_time = parse_double(value, "final_time");
  else if (key == "gamma_q") parameters.reduction_damping = parse_double(value, "gamma_q");
  else if (key == "dissipation") parameters.dissipation = parse_double(value, "dissipation");
  else if (key == "amplitude") parameters.pulse_amplitude = parse_double(value, "amplitude");
  else if (key == "pulse_center") parameters.pulse_center_fraction = parse_double(value, "pulse_center");
  else if (key == "pulse_width") parameters.pulse_width_fraction = parse_double(value, "pulse_width");
  else if (key == "steps") parameters.steps = parse_int(value, "steps");
  else if (key == "diagnostic_every") parameters.diagnostic_interval = parse_int(value, "diagnostic_every");
  else if (key == "checkpoint_every") parameters.checkpoint_interval = parse_int(value, "checkpoint_every");
  else if (key == "modes") parameters.modes = parse_modes(value);
  else if (key == "output") parameters.output_directory = std::string(value);
  else throw std::invalid_argument(std::string("unknown parameter: ") + std::string(key));
}

inline void validate(const Parameters& parameters) {
  if (parameters.mass <= 0.0) throw std::invalid_argument("mass must be positive");
  if (std::abs(parameters.spin) > parameters.mass)
    throw std::invalid_argument("absolute spin must not exceed mass");
  if (parameters.compactification_scale <= 0.0)
    throw std::invalid_argument("L must be positive");
  if (parameters.radial_points < 9) throw std::invalid_argument("nr must be at least 9");
  if (parameters.ell_max < 3) throw std::invalid_argument("ellmax must be at least 3");
  const int exact_product_nodes = (3 * parameters.ell_max + 2) / 2;
  if (parameters.theta_points <
      std::max(parameters.ell_max + 1, exact_product_nodes)) {
    throw std::invalid_argument(
        "ntheta is too small for the retained nonlinear angular band");
  }
  if (parameters.cfl <= 0.0 || parameters.final_time < 0.0)
    throw std::invalid_argument("cfl must be positive and final_time nonnegative");
  if (parameters.steps <= 0) throw std::invalid_argument("steps must be positive");
  if (parameters.diagnostic_interval <= 0)
    throw std::invalid_argument("diagnostic_every must be positive");
  if (parameters.checkpoint_interval < 0)
    throw std::invalid_argument("checkpoint_every must be nonnegative");
  if (!std::isfinite(parameters.pulse_amplitude) ||
      parameters.pulse_width_fraction <= 0.0 ||
      parameters.pulse_center_fraction < 0.0 ||
      parameters.pulse_center_fraction > 1.0) {
    throw std::invalid_argument("invalid pulse parameters");
  }
  if (parameters.modes.empty()) throw std::invalid_argument("modes must not be empty");
  std::vector<int> sorted_modes = parameters.modes;
  std::sort(sorted_modes.begin(), sorted_modes.end());
  if (std::adjacent_find(sorted_modes.begin(), sorted_modes.end()) !=
      sorted_modes.end()) {
    throw std::invalid_argument("modes must be unique");
  }
  for (const int mode : sorted_modes) {
    if (std::abs(mode) > parameters.ell_max) {
      throw std::invalid_argument("absolute m must not exceed ellmax");
    }
    if (!std::binary_search(sorted_modes.begin(), sorted_modes.end(), -mode)) {
      throw std::invalid_argument("modes must be closed under m -> -m");
    }
  }
  if (parameters.seed_ell < std::max(2, std::abs(parameters.seed_mode)) ||
      parameters.seed_ell > parameters.ell_max ||
      !std::binary_search(sorted_modes.begin(), sorted_modes.end(),
                          parameters.seed_mode)) {
    throw std::invalid_argument("seed (ell,m) is outside the stored band");
  }
  if (parameters.output_directory.find_first_of("\r\n") !=
      std::string::npos) {
    throw std::invalid_argument("output path contains a newline");
  }
}

}  // namespace teuk
