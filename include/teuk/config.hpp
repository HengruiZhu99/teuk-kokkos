#pragma once

#include <charconv>
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
  double cfl = 0.1;
  double final_time = 1.0;
  double reduction_damping = 0.1;
  double dissipation = 0.01;
  std::vector<int> modes{-2, 2};
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
  else if (key == "cfl") parameters.cfl = parse_double(value, "cfl");
  else if (key == "final_time") parameters.final_time = parse_double(value, "final_time");
  else if (key == "gamma_q") parameters.reduction_damping = parse_double(value, "gamma_q");
  else if (key == "dissipation") parameters.dissipation = parse_double(value, "dissipation");
  else throw std::invalid_argument(std::string("unknown parameter: ") + std::string(key));
}

inline void validate(const Parameters& parameters) {
  if (parameters.mass <= 0.0) throw std::invalid_argument("mass must be positive");
  if (std::abs(parameters.spin) > parameters.mass)
    throw std::invalid_argument("absolute spin must not exceed mass");
  if (parameters.compactification_scale <= 0.0)
    throw std::invalid_argument("L must be positive");
  if (parameters.radial_points < 9) throw std::invalid_argument("nr must be at least 9");
  if (parameters.theta_points < 4) throw std::invalid_argument("ntheta must be at least 4");
  if (parameters.cfl <= 0.0 || parameters.final_time < 0.0)
    throw std::invalid_argument("cfl must be positive and final_time nonnegative");
}

}  // namespace teuk
