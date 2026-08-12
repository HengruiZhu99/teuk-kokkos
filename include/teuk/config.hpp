#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "teuk/modes.hpp"
#include "teuk/run_parameters.hpp"

namespace teuk {

namespace config_detail {

inline std::string trim(std::string_view text) {
  const auto whitespace = [](const unsigned char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
  };
  std::size_t begin = 0;
  while (begin < text.size() && whitespace(text[begin])) ++begin;
  std::size_t end = text.size();
  while (end > begin && whitespace(text[end - 1])) --end;
  return std::string(text.substr(begin, end - begin));
}

inline std::pair<std::string, std::string> parse_assignment(
    const std::string_view text, const std::string& context) {
  const std::size_t delimiter = text.find('=');
  if (delimiter == std::string_view::npos ||
      text.find('=', delimiter + 1) != std::string_view::npos) {
    throw std::invalid_argument(context + ": expected exactly one key = value");
  }
  std::string key = trim(text.substr(0, delimiter));
  std::string value = trim(text.substr(delimiter + 1));
  if (key.empty() || value.empty()) {
    throw std::invalid_argument(context + ": empty configuration key or value");
  }
  return {std::move(key), std::move(value)};
}

template <class Value>
Value parse_number(const std::string& text, const std::string& key) {
  Value value{};
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument("key '" + key + "' has malformed value '" +
                                text + "'");
  }
  return value;
}

inline bool parse_bool(const std::string& text, const std::string& key) {
  if (text == "true" || text == "1") return true;
  if (text == "false" || text == "0") return false;
  throw std::invalid_argument("key '" + key +
                              "' requires true, false, 1, or 0");
}

inline std::vector<int> parse_int_list(const std::string& text,
                                       const std::string& key) {
  std::vector<int> values;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t delimiter = text.find(',', begin);
    const std::size_t end =
        delimiter == std::string::npos ? text.size() : delimiter;
    const std::string token = trim(
        std::string_view(text).substr(begin, end - begin));
    if (token.empty()) {
      throw std::invalid_argument("key '" + key + "' contains an empty item");
    }
    values.push_back(parse_number<int>(token, key));
    if (delimiter == std::string::npos) break;
    begin = delimiter + 1;
  }
  if (values.empty()) {
    throw std::invalid_argument("key '" + key + "' requires a nonempty list");
  }
  return values;
}

using EntryMap = std::map<std::string, std::string>;

inline EntryMap parse_config_stream(std::istream& input,
                                    const std::string& source) {
  EntryMap entries;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    if (trim(line).empty()) continue;
    auto [key, value] = parse_assignment(
        line, source + ":" + std::to_string(line_number));
    if (!entries.emplace(key, value).second) {
      throw std::invalid_argument(source + ":" +
                                  std::to_string(line_number) +
                                  ": duplicate key '" + key + "'");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading configuration " + source);
  }
  return entries;
}

inline EntryMap parse_overrides(const std::vector<std::string>& overrides) {
  EntryMap entries;
  for (std::size_t i = 0; i < overrides.size(); ++i) {
    auto [key, value] = parse_assignment(
        overrides[i], "command-line override " + std::to_string(i + 1));
    if (!entries.emplace(key, value).second) {
      throw std::invalid_argument("duplicate command-line override '" + key +
                                  "'");
    }
  }
  return entries;
}

inline void merge_overrides(EntryMap& entries, const EntryMap& overrides) {
  for (const auto& [key, value] : overrides) entries[key] = value;
}

struct InitialModeBuilder {
  InitialMode mode;
  bool ell = false;
  bool m = false;
  bool real = false;
  bool imag = false;
};

inline bool parse_mode_field(const std::string& key, std::size_t& index,
                             std::string& field) {
  constexpr std::string_view prefix = "initial_data.mode.";
  if (!key.starts_with(prefix)) return false;
  const std::size_t index_begin = prefix.size();
  const std::size_t delimiter = key.find('.', index_begin);
  if (delimiter == std::string::npos) return false;
  const std::string index_text = key.substr(index_begin, delimiter - index_begin);
  if (index_text.empty()) return false;
  index = parse_number<std::size_t>(index_text, key);
  field = key.substr(delimiter + 1);
  return field == "ell" || field == "m" || field == "amplitude_real" ||
         field == "amplitude_imag";
}

inline void require_finite(const double value, const std::string& key) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("key '" + key + "' must be finite");
  }
}

inline std::string source_mode_name(const SecondOrderSourceMode mode) {
  switch (mode) {
    case SecondOrderSourceMode::Disabled: return "disabled";
    case SecondOrderSourceMode::ConstraintAware: return "constraint_aware";
    case SecondOrderSourceMode::Unrestricted: return "unrestricted";
  }
  throw std::invalid_argument("unsupported source mode");
}

inline std::string reduction_name(const ReductionEvolution reduction) {
  switch (reduction) {
    case ReductionEvolution::FreeDamped: return "free_damped";
    case ReductionEvolution::StageConstrained: return "stage_constrained";
  }
  throw std::invalid_argument("unsupported reduction mode");
}

inline std::string initial_data_name(const InitialDataType type) {
  switch (type) {
    case InitialDataType::Gaussian: return "gaussian";
    case InitialDataType::Checkpoint: return "checkpoint";
  }
  throw std::invalid_argument("unsupported initial-data type");
}

}  // namespace config_detail

inline RunParameters resolve_run_parameters(
    const config_detail::EntryMap& entries) {
  RunParameters parameters;
  std::map<std::size_t, config_detail::InitialModeBuilder> mode_builders;
  mode_builders[0].mode = parameters.initial_data.modes.front();
  std::set<std::string> recognized;
  const auto value = [&](const std::string& key) -> const std::string* {
    const auto found = entries.find(key);
    if (found == entries.end()) return nullptr;
    recognized.insert(key);
    return &found->second;
  };
  const auto set_double = [&](const std::string& key, double& destination) {
    if (const auto* text = value(key)) {
      destination = config_detail::parse_number<double>(*text, key);
    }
  };
  const auto set_int = [&](const std::string& key, int& destination) {
    if (const auto* text = value(key)) {
      destination = config_detail::parse_number<int>(*text, key);
    }
  };
  const auto set_bool = [&](const std::string& key, bool& destination) {
    if (const auto* text = value(key)) {
      destination = config_detail::parse_bool(*text, key);
    }
  };
  const auto set_string = [&](const std::string& key,
                              std::string& destination) {
    if (const auto* text = value(key)) destination = *text;
  };

  set_int("config_version", parameters.config_version);
  set_double("mass", parameters.background.mass);
  set_double("spin", parameters.background.spin);
  set_double("compactification_length",
             parameters.background.compactification_length);
  set_int("nr", parameters.grid.radial_points);
  set_int("ntheta", parameters.grid.theta_points);
  set_int("ellmax_first", parameters.grid.ell_max_first);
  const bool second_ell_explicit = entries.contains("ellmax_second");
  set_int("ellmax_second", parameters.grid.ell_max_second);
  if (!second_ell_explicit) {
    parameters.grid.ell_max_second = parameters.grid.ell_max_first;
  }
  if (const auto* text = value("first_order_modes")) {
    parameters.grid.first_order_modes =
        config_detail::parse_int_list(*text, "first_order_modes");
  }
  if (const auto* text = value("second_order_modes")) {
    parameters.grid.second_order_modes =
        config_detail::parse_int_list(*text, "second_order_modes");
  }
  set_double("final_time", parameters.time.final_time);
  set_int("steps", parameters.time.steps);
  set_double("cfl", parameters.time.cfl);
  set_double("reduction_damping", parameters.method.reduction_damping);
  set_double("dissipation", parameters.method.dissipation);
  if (const auto* text = value("radial_discretization")) {
    parameters.method.radial_discretization =
        parse_radial_discretization(*text);
  }
  if (const auto* text = value("reduction_mode")) {
    if (*text == "free_damped") {
      parameters.method.reduction = ReductionEvolution::FreeDamped;
    } else if (*text == "stage_constrained") {
      parameters.method.reduction = ReductionEvolution::StageConstrained;
    } else {
      throw std::invalid_argument(
          "key 'reduction_mode' requires free_damped or stage_constrained");
    }
  }
  if (const auto* text = value("initial_data.type")) {
    if (*text == "gaussian") {
      parameters.initial_data.type = InitialDataType::Gaussian;
    } else if (*text == "checkpoint") {
      parameters.initial_data.type = InitialDataType::Checkpoint;
    } else {
      throw std::invalid_argument(
          "key 'initial_data.type' requires gaussian or checkpoint");
    }
  }
  set_bool("initial_data.add_sharp_partner",
           parameters.initial_data.add_sharp_partner);
  set_bool("initial_data.compact_support",
           parameters.initial_data.compact_support);
  set_double("initial_data.center", parameters.initial_data.center_fraction);
  set_double("initial_data.width", parameters.initial_data.width_fraction);
  set_string("initial_data.time_derivative",
             parameters.initial_data.time_derivative);
  set_string("initial_data.checkpoint_directory",
             parameters.initial_data.checkpoint_directory);
  if (parameters.initial_data.checkpoint_directory == "none") {
    parameters.initial_data.checkpoint_directory.clear();
  }

  auto& base = mode_builders[0];
  if (const auto* text = value("initial_data.seed_ell")) {
    base.mode.ell = config_detail::parse_number<int>(*text,
                                                    "initial_data.seed_ell");
  }
  if (const auto* text = value("initial_data.seed_m")) {
    base.mode.m =
        config_detail::parse_number<int>(*text, "initial_data.seed_m");
  }
  double base_real = base.mode.amplitude.real();
  double base_imag = base.mode.amplitude.imag();
  if (const auto* text = value("initial_data.amplitude_real")) {
    base_real = config_detail::parse_number<double>(
        *text, "initial_data.amplitude_real");
  }
  if (const auto* text = value("initial_data.amplitude_imag")) {
    base_imag = config_detail::parse_number<double>(
        *text, "initial_data.amplitude_imag");
  }
  base.mode.amplitude = Complex(base_real, base_imag);

  for (const auto& [key, text] : entries) {
    std::size_t index = 0;
    std::string field;
    if (!config_detail::parse_mode_field(key, index, field)) continue;
    recognized.insert(key);
    auto& builder = mode_builders[index];
    if (field == "ell") {
      builder.mode.ell = config_detail::parse_number<int>(text, key);
      builder.ell = true;
    } else if (field == "m") {
      builder.mode.m = config_detail::parse_number<int>(text, key);
      builder.m = true;
    } else if (field == "amplitude_real") {
      builder.mode.amplitude = Complex(
          config_detail::parse_number<double>(text, key),
          builder.mode.amplitude.imag());
      builder.real = true;
    } else if (field == "amplitude_imag") {
      builder.mode.amplitude = Complex(
          builder.mode.amplitude.real(),
          config_detail::parse_number<double>(text, key));
      builder.imag = true;
    }
  }
  parameters.initial_data.modes.clear();
  for (const auto& [index, builder] : mode_builders) {
    if (index != 0 &&
        !(builder.ell && builder.m && builder.real && builder.imag)) {
      throw std::invalid_argument(
          "initial_data.mode." + std::to_string(index) +
          " requires ell, m, amplitude_real, and amplitude_imag");
    }
    parameters.initial_data.modes.push_back(builder.mode);
  }

  set_bool("second_order.enabled", parameters.second_order.enabled);
  if (const auto* text = value("second_order.source_mode")) {
    if (*text == "constraint_aware") {
      parameters.second_order.source_mode =
          SecondOrderSourceMode::ConstraintAware;
    } else if (*text == "unrestricted") {
      parameters.second_order.source_mode = SecondOrderSourceMode::Unrestricted;
    } else {
      throw std::invalid_argument(
          "key 'second_order.source_mode' requires constraint_aware or unrestricted");
    }
  }
  set_double("second_order.source_start_time",
             parameters.second_order.source_start_time);
  set_double("second_order.constraint_tolerance",
             parameters.second_order.normalized_constraint_tolerance);
  set_int("second_order.required_consecutive_passes",
          parameters.second_order.required_consecutive_passes);
  set_bool("second_order.allow_truncated_daughter_modes",
           parameters.second_order.allow_truncated_daughter_modes);

  set_bool("plus2.enabled", parameters.plus2.enabled);
  if (const auto* text = value("plus2.mode")) {
    parameters.plus2.mode = parse_plus2_run_mode(*text);
  }
  if (const auto* text = value("plus2.linear.method")) {
    parameters.plus2.linear_method = parse_plus2_linear_method(*text);
  }
  set_bool("plus2.linear.evolve_validation",
           parameters.plus2.linear_evolve_validation);
  if (const auto* text = value("plus2.second.method")) {
    parameters.plus2.second_method = parse_plus2_second_method(*text);
  }
  if (const auto* text = value("plus2.second.initial_policy")) {
    parameters.plus2.second_initial_policy =
        parse_plus2_initial_policy(*text);
  }
  set_string("plus2.second.checkpoint", parameters.plus2.second_checkpoint);
  if (parameters.plus2.second_checkpoint == "none") {
    parameters.plus2.second_checkpoint.clear();
  }
  const bool plus2_first_ell_explicit =
      entries.contains("plus2.ell_max_first");
  const bool plus2_second_ell_explicit =
      entries.contains("plus2.ell_max_second");
  set_int("plus2.ell_max_first", parameters.plus2.ell_max_first);
  set_int("plus2.ell_max_second", parameters.plus2.ell_max_second);
  if (!plus2_first_ell_explicit) {
    parameters.plus2.ell_max_first = parameters.grid.ell_max_first;
  }
  if (!plus2_second_ell_explicit) {
    parameters.plus2.ell_max_second = parameters.grid.ell_max_second;
  }
  set_bool("plus2.output.regularized", parameters.plus2.output.regularized);
  set_bool("plus2.output.physical_tetrad_field",
           parameters.plus2.output.physical_tetrad_field);
  set_bool("plus2.output.source_families",
           parameters.plus2.output.source_families);
  set_bool("plus2.output.ordered_pairs",
           parameters.plus2.output.ordered_pairs);

  set_string("output.directory", parameters.output.directory);
  set_int("output.diagnostic_every",
          parameters.output.diagnostic_interval);
  set_int("output.checkpoint_every",
          parameters.output.checkpoint_interval);

  for (const auto& [key, ignored] : entries) {
    (void)ignored;
    if (!recognized.contains(key)) {
      throw std::invalid_argument("unknown configuration key '" + key + "'");
    }
  }
  return parameters;
}

inline void validate_run_parameters(const RunParameters& parameters) {
  if (parameters.config_version != runtime_config_schema_version) {
    throw std::invalid_argument("config_version must be 1");
  }
  config_detail::require_finite(parameters.background.mass, "mass");
  config_detail::require_finite(parameters.background.spin, "spin");
  config_detail::require_finite(parameters.background.compactification_length,
                                "compactification_length");
  if (!(parameters.background.mass > 0.0)) {
    throw std::invalid_argument("mass must be positive");
  }
  if (std::abs(parameters.background.spin) > parameters.background.mass) {
    throw std::invalid_argument("absolute spin must not exceed mass");
  }
  if (!(parameters.background.compactification_length > 0.0)) {
    throw std::invalid_argument("compactification_length must be positive");
  }
  if (parameters.grid.radial_points < static_cast<int>(
          radial_minimum_points(parameters.method.radial_discretization))) {
    throw std::invalid_argument(
        "nr is too small for radial_discretization=" +
        std::string(radial_discretization_name(
            parameters.method.radial_discretization)));
  }
  if (parameters.grid.ell_max_first < 3 ||
      parameters.grid.ell_max_second < 3) {
    throw std::invalid_argument("ellmax_first and ellmax_second must be at least 3");
  }
  const int product_nodes =
      (2 * parameters.grid.ell_max_first +
       parameters.grid.ell_max_second + 2) /
      2;
  if (parameters.grid.theta_points <
      std::max({parameters.grid.ell_max_first + 1,
                parameters.grid.ell_max_second + 1, product_nodes})) {
    throw std::invalid_argument(
        "ntheta is too small for the selected first/second angular bands");
  }
  const auto validate_modes = [](const std::vector<int>& modes,
                                 const int ell_max,
                                 const std::string& key) {
    if (modes.empty()) {
      throw std::invalid_argument(key + " must not be empty");
    }
    std::vector<int> sorted = modes;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      throw std::invalid_argument(key + " must contain unique signed modes");
    }
    for (const int m : sorted) {
      if (std::abs(m) > ell_max) {
        throw std::invalid_argument(key + " contains |m| above its ellmax");
      }
      if (!std::binary_search(sorted.begin(), sorted.end(), -m)) {
        throw std::invalid_argument(key + " must be closed under m -> -m");
      }
    }
  };
  validate_modes(parameters.grid.first_order_modes,
                 parameters.grid.ell_max_first, "first_order_modes");
  validate_modes(parameters.grid.second_order_modes,
                 parameters.grid.ell_max_second, "second_order_modes");
  if (parameters.second_order.enabled) {
    (void)validate_quadratic_daughter_modes(
        parameters.grid.first_order_modes,
        parameters.grid.second_order_modes,
        parameters.second_order.allow_truncated_daughter_modes);
  }
  config_detail::require_finite(parameters.time.final_time, "final_time");
  config_detail::require_finite(parameters.time.cfl, "cfl");
  if (!(parameters.time.final_time > 0.0) || parameters.time.steps <= 0 ||
      !(parameters.time.cfl > 0.0)) {
    throw std::invalid_argument("final_time, steps, and cfl must be positive");
  }
  config_detail::require_finite(parameters.method.reduction_damping,
                                "reduction_damping");
  config_detail::require_finite(parameters.method.dissipation, "dissipation");
  if (parameters.method.reduction_damping < 0.0 ||
      parameters.method.dissipation < 0.0) {
    throw std::invalid_argument(
        "reduction_damping and dissipation must be nonnegative");
  }
  const double horizon_radius =
      parameters.background.mass +
      std::sqrt(parameters.background.mass * parameters.background.mass -
                parameters.background.spin * parameters.background.spin);
  const double compact_horizon =
      parameters.background.compactification_length *
      parameters.background.compactification_length / horizon_radius;
  const double radial_spacing =
      compact_horizon /
      static_cast<double>(parameters.grid.radial_points - 1);
  const double time_step =
      parameters.time.final_time / static_cast<double>(parameters.time.steps);
  if (!radial_dissipation_rk4_step_is_admissible(
          parameters.method.radial_discretization,
          parameters.method.dissipation, time_step / radial_spacing)) {
    throw std::invalid_argument(
        "steps are too small for the selected radial dissipation RK4 bound");
  }
  if (parameters.initial_data.type == InitialDataType::Gaussian) {
    if (parameters.initial_data.modes.empty()) {
      throw std::invalid_argument("Gaussian initial data require a seed mode");
    }
    std::set<std::pair<int, int>> seed_keys;
    for (const auto& mode : parameters.initial_data.modes) {
      config_detail::require_finite(mode.amplitude.real(),
                                    "initial_data amplitude_real");
      config_detail::require_finite(mode.amplitude.imag(),
                                    "initial_data amplitude_imag");
      if (mode.ell < std::max(2, std::abs(mode.m)) ||
          mode.ell > parameters.grid.ell_max_first) {
        throw std::invalid_argument(
            "initial-data seed ell exceeds ellmax_first or minimum ell");
      }
      if (std::find(parameters.grid.first_order_modes.begin(),
                    parameters.grid.first_order_modes.end(), mode.m) ==
          parameters.grid.first_order_modes.end()) {
        throw std::invalid_argument(
            "initial-data seed m is absent from first_order_modes");
      }
      if (!seed_keys.emplace(mode.ell, mode.m).second) {
        throw std::invalid_argument("duplicate initial-data (ell,m) seed");
      }
      if (parameters.initial_data.add_sharp_partner && mode.m != 0 &&
          seed_keys.contains({mode.ell, -mode.m})) {
        throw std::invalid_argument(
            "add_sharp_partner would duplicate an explicit seed");
      }
    }
    config_detail::require_finite(parameters.initial_data.center_fraction,
                                  "initial_data.center");
    config_detail::require_finite(parameters.initial_data.width_fraction,
                                  "initial_data.width");
    if (parameters.initial_data.center_fraction < 0.0 ||
        parameters.initial_data.center_fraction > 1.0 ||
        !(parameters.initial_data.width_fraction > 0.0)) {
      throw std::invalid_argument(
          "initial_data.center must be in [0,1] and width must be positive");
    }
    if (parameters.initial_data.time_derivative != "zero") {
      throw std::invalid_argument(
          "initial_data.time_derivative currently supports only 'zero'");
    }
  } else if (parameters.initial_data.checkpoint_directory.empty()) {
    throw std::invalid_argument(
        "checkpoint initial data require initial_data.checkpoint_directory");
  }
  config_detail::require_finite(parameters.second_order.source_start_time,
                                "second_order.source_start_time");
  config_detail::require_finite(
      parameters.second_order.normalized_constraint_tolerance,
      "second_order.constraint_tolerance");
  if (parameters.second_order.source_start_time < 0.0 ||
      parameters.second_order.normalized_constraint_tolerance < 0.0 ||
      parameters.second_order.required_consecutive_passes < 1) {
    throw std::invalid_argument("invalid second-order source activation policy");
  }

  const bool plus2_mode_enabled =
      parameters.plus2.mode != Plus2RunMode::Disabled;
  (void)plus2_run_mode_name(parameters.plus2.mode);
  (void)plus2_linear_method_name(parameters.plus2.linear_method);
  (void)plus2_second_method_name(parameters.plus2.second_method);
  (void)plus2_initial_policy_name(
      parameters.plus2.second_initial_policy);
  if (parameters.plus2.enabled != plus2_mode_enabled) {
    throw std::invalid_argument(
        "plus2.enabled must be true exactly when plus2.mode is not disabled");
  }
  if (parameters.plus2.ell_max_first < 2 ||
      parameters.plus2.ell_max_second < 2) {
    throw std::invalid_argument(
        "plus2 ell_max_first and ell_max_second must be at least 2");
  }
  const auto validate_plus2_registry_band = [](const std::vector<int>& modes,
                                                const int ell_max,
                                                const char* label) {
    for (const int mode : modes) {
      if (std::abs(mode) > ell_max) {
        throw std::invalid_argument(std::string(label) +
                                    " contains |m| above the plus2 band");
      }
    }
  };
  validate_plus2_registry_band(parameters.grid.first_order_modes,
                               parameters.plus2.ell_max_first,
                               "first_order_modes");
  validate_plus2_registry_band(parameters.grid.second_order_modes,
                               parameters.plus2.ell_max_second,
                               "second_order_modes");
  const int plus2_product_nodes =
      (2 * parameters.plus2.ell_max_first +
       parameters.plus2.ell_max_second + 2) /
      2;
  if (parameters.plus2.enabled &&
      parameters.grid.theta_points <
          std::max({parameters.plus2.ell_max_first + 1,
                    parameters.plus2.ell_max_second + 1,
                    plus2_product_nodes})) {
    throw std::invalid_argument(
        "ntheta is too small for the selected plus2 angular bands");
  }
  const bool checkpoint_policy =
      parameters.plus2.second_initial_policy ==
      Plus2InitialPolicy::Checkpoint;
  if (checkpoint_policy != !parameters.plus2.second_checkpoint.empty()) {
    throw std::invalid_argument(
        "plus2.second.checkpoint is required exactly for checkpoint initial policy");
  }
  if (!parameters.plus2.second_checkpoint.empty() &&
      parameters.plus2.second_checkpoint.find_first_of("\r\n#=") !=
          std::string::npos) {
    throw std::invalid_argument(
        "plus2.second.checkpoint must be a config-safe single-line path");
  }
  if ((parameters.plus2.mode == Plus2RunMode::Disabled ||
       parameters.plus2.mode == Plus2RunMode::DiagnosticOnly) &&
      checkpoint_policy) {
    throw std::invalid_argument(
        "plus2 disabled/diagnostic_only mode cannot load a second-order companion checkpoint");
  }
  if ((parameters.plus2.mode == Plus2RunMode::Concurrent ||
       parameters.plus2.mode == Plus2RunMode::Replay) &&
      !parameters.second_order.enabled) {
    throw std::invalid_argument(
        "plus2 concurrent/replay requires second_order.enabled=true for four-field evolution");
  }
  if (parameters.plus2.output.ordered_pairs &&
      !parameters.plus2.output.source_families) {
    throw std::invalid_argument(
        "plus2 ordered-pair output requires source-family output");
  }
  if (parameters.plus2.enabled &&
      !parameters.plus2.output.regularized &&
      !parameters.plus2.output.physical_tetrad_field) {
    throw std::invalid_argument(
        "enabled plus2 mode must output a regularized or physical tetrad field");
  }
  if (parameters.output.directory.empty() ||
      parameters.output.directory.find_first_of("\r\n") != std::string::npos) {
    throw std::invalid_argument("output.directory must be a nonempty single line");
  }
  if (parameters.output.diagnostic_interval <= 0 ||
      parameters.output.checkpoint_interval < 0) {
    throw std::invalid_argument(
        "diagnostic cadence must be positive and checkpoint cadence nonnegative");
  }
}

inline RunParameters parse_run_configuration(
    const std::filesystem::path& path,
    const std::vector<std::string>& overrides = {}) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open configuration file: " + path.string());
  }
  auto entries = config_detail::parse_config_stream(input, path.string());
  config_detail::merge_overrides(entries,
                                 config_detail::parse_overrides(overrides));
  RunParameters parameters = resolve_run_parameters(entries);
  validate_run_parameters(parameters);
  return parameters;
}

inline RunParameters parse_run_configuration_text(
    const std::string& text,
    const std::vector<std::string>& overrides = {}) {
  std::istringstream input(text);
  auto entries = config_detail::parse_config_stream(input, "<configuration>");
  config_detail::merge_overrides(entries,
                                 config_detail::parse_overrides(overrides));
  RunParameters parameters = resolve_run_parameters(entries);
  validate_run_parameters(parameters);
  return parameters;
}

inline std::string resolved_configuration_text(
    const RunParameters& parameters) {
  validate_run_parameters(parameters);
  std::ostringstream output;
  output << std::setprecision(17) << std::boolalpha
         << "config_version = " << parameters.config_version << '\n'
         << "mass = " << parameters.background.mass << '\n'
         << "spin = " << parameters.background.spin << '\n'
         << "compactification_length = "
         << parameters.background.compactification_length << '\n'
         << "nr = " << parameters.grid.radial_points << '\n'
         << "ntheta = " << parameters.grid.theta_points << '\n'
         << "ellmax_first = " << parameters.grid.ell_max_first << '\n'
         << "ellmax_second = " << parameters.grid.ell_max_second << '\n';
  const auto write_modes = [&](const char* key,
                               const std::vector<int>& modes) {
    output << key << " = ";
    for (std::size_t i = 0; i < modes.size(); ++i) {
      if (i != 0) output << ',';
      output << modes[i];
    }
    output << '\n';
  };
  write_modes("first_order_modes", parameters.grid.first_order_modes);
  write_modes("second_order_modes", parameters.grid.second_order_modes);
  output << "final_time = " << parameters.time.final_time << '\n'
         << "steps = " << parameters.time.steps << '\n'
         << "cfl = " << parameters.time.cfl << '\n'
         << "reduction_damping = " << parameters.method.reduction_damping
         << '\n'
         << "dissipation = " << parameters.method.dissipation << '\n'
         << "radial_discretization = "
         << radial_discretization_name(
                parameters.method.radial_discretization)
         << '\n'
         << "reduction_mode = "
         << config_detail::reduction_name(parameters.method.reduction) << '\n'
         << "initial_data.type = "
         << config_detail::initial_data_name(parameters.initial_data.type)
         << '\n'
         << "initial_data.add_sharp_partner = "
         << parameters.initial_data.add_sharp_partner << '\n'
         << "initial_data.compact_support = "
         << parameters.initial_data.compact_support << '\n'
         << "initial_data.center = "
         << parameters.initial_data.center_fraction << '\n'
         << "initial_data.width = "
         << parameters.initial_data.width_fraction << '\n'
         << "initial_data.time_derivative = "
         << parameters.initial_data.time_derivative << '\n'
         << "initial_data.checkpoint_directory = "
         << (parameters.initial_data.checkpoint_directory.empty()
                 ? "none"
                 : parameters.initial_data.checkpoint_directory)
         << '\n';
  for (std::size_t i = 0; i < parameters.initial_data.modes.size(); ++i) {
    const auto& mode = parameters.initial_data.modes[i];
    output << "initial_data.mode." << i << ".ell = " << mode.ell << '\n'
           << "initial_data.mode." << i << ".m = " << mode.m << '\n'
           << "initial_data.mode." << i << ".amplitude_real = "
           << mode.amplitude.real() << '\n'
           << "initial_data.mode." << i << ".amplitude_imag = "
           << mode.amplitude.imag() << '\n';
  }
  output << "second_order.enabled = " << parameters.second_order.enabled << '\n'
         << "second_order.source_mode = "
         << config_detail::source_mode_name(parameters.second_order.source_mode)
         << '\n'
         << "second_order.source_start_time = "
         << parameters.second_order.source_start_time << '\n'
         << "second_order.constraint_tolerance = "
         << parameters.second_order.normalized_constraint_tolerance << '\n'
         << "second_order.required_consecutive_passes = "
         << parameters.second_order.required_consecutive_passes << '\n'
         << "second_order.allow_truncated_daughter_modes = "
         << parameters.second_order.allow_truncated_daughter_modes << '\n'
         << "plus2.enabled = " << parameters.plus2.enabled << '\n'
         << "plus2.mode = " << plus2_run_mode_name(parameters.plus2.mode)
         << '\n'
         << "plus2.linear.method = "
         << plus2_linear_method_name(parameters.plus2.linear_method)
         << '\n'
         << "plus2.linear.evolve_validation = "
         << parameters.plus2.linear_evolve_validation << '\n'
         << "plus2.second.method = "
         << plus2_second_method_name(parameters.plus2.second_method)
         << '\n'
         << "plus2.second.initial_policy = "
         << plus2_initial_policy_name(
                parameters.plus2.second_initial_policy)
         << '\n'
         << "plus2.second.checkpoint = "
         << (parameters.plus2.second_checkpoint.empty()
                 ? "none"
                 : parameters.plus2.second_checkpoint)
         << '\n'
         << "plus2.ell_max_first = " << parameters.plus2.ell_max_first
         << '\n'
         << "plus2.ell_max_second = " << parameters.plus2.ell_max_second
         << '\n'
         << "plus2.output.regularized = "
         << parameters.plus2.output.regularized << '\n'
         << "plus2.output.physical_tetrad_field = "
         << parameters.plus2.output.physical_tetrad_field << '\n'
         << "plus2.output.source_families = "
         << parameters.plus2.output.source_families << '\n'
         << "plus2.output.ordered_pairs = "
         << parameters.plus2.output.ordered_pairs << '\n'
         << "output.directory = " << parameters.output.directory << '\n'
         << "output.diagnostic_every = "
         << parameters.output.diagnostic_interval << '\n'
         << "output.checkpoint_every = "
         << parameters.output.checkpoint_interval << '\n';
  return output.str();
}

}  // namespace teuk
