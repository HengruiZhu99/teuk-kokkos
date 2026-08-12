#pragma once

#include <stdexcept>
#include <string>

namespace teuk {

// Dependency-neutral runtime and provenance enums shared by configuration,
// replay, and companion checkpoints.
enum class Plus2RunMode { Disabled, DiagnosticOnly, Concurrent, Replay };
enum class Plus2InitialPolicy { Zero, Checkpoint };
enum class Plus2LinearMethod { MetricCurvature, Tsi, Both };
enum class Plus2SecondMethod { SourcedCompanion };

inline const char* plus2_run_mode_name(const Plus2RunMode mode) {
  switch (mode) {
    case Plus2RunMode::Disabled: return "disabled";
    case Plus2RunMode::DiagnosticOnly: return "diagnostic_only";
    case Plus2RunMode::Concurrent: return "concurrent";
    case Plus2RunMode::Replay: return "replay";
  }
  throw std::invalid_argument("unsupported plus2 run mode");
}

inline Plus2RunMode parse_plus2_run_mode(const std::string& text) {
  if (text == "disabled") return Plus2RunMode::Disabled;
  if (text == "diagnostic_only") return Plus2RunMode::DiagnosticOnly;
  if (text == "concurrent") return Plus2RunMode::Concurrent;
  if (text == "replay") return Plus2RunMode::Replay;
  throw std::invalid_argument("unknown plus2 run mode: " + text);
}

inline const char* plus2_initial_policy_name(const Plus2InitialPolicy policy) {
  switch (policy) {
    case Plus2InitialPolicy::Zero: return "zero";
    case Plus2InitialPolicy::Checkpoint: return "checkpoint";
  }
  throw std::invalid_argument("unsupported plus2 initial policy");
}

inline Plus2InitialPolicy parse_plus2_initial_policy(const std::string& text) {
  if (text == "zero") return Plus2InitialPolicy::Zero;
  if (text == "checkpoint") return Plus2InitialPolicy::Checkpoint;
  throw std::invalid_argument("unknown plus2 initial policy: " + text);
}

inline const char* plus2_linear_method_name(const Plus2LinearMethod method) {
  switch (method) {
    case Plus2LinearMethod::MetricCurvature: return "metric_curvature";
    case Plus2LinearMethod::Tsi: return "tsi";
    case Plus2LinearMethod::Both: return "both";
  }
  throw std::invalid_argument("unsupported plus2 linear method");
}

inline Plus2LinearMethod parse_plus2_linear_method(const std::string& text) {
  if (text == "metric_curvature") return Plus2LinearMethod::MetricCurvature;
  if (text == "tsi") return Plus2LinearMethod::Tsi;
  if (text == "both") return Plus2LinearMethod::Both;
  throw std::invalid_argument("unknown plus2 linear method: " + text);
}

inline const char* plus2_second_method_name(const Plus2SecondMethod method) {
  switch (method) {
    case Plus2SecondMethod::SourcedCompanion: return "sourced_companion";
  }
  throw std::invalid_argument("unsupported plus2 second method");
}

inline Plus2SecondMethod parse_plus2_second_method(const std::string& text) {
  if (text == "sourced_companion") {
    return Plus2SecondMethod::SourcedCompanion;
  }
  throw std::invalid_argument("unknown plus2 second method: " + text);
}

}  // namespace teuk
