#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Angular metadata for the complete evolved state. Reconstruction fields are
// first-order metric variables and therefore use the first-order bandlimit;
// the final Teukolsky triple may use a distinct second-order bandlimit.
inline constexpr std::array<int, point_pipeline_field_count>
    pipeline_field_spins{-2, -2, -2, -1, -2, 0, -2,
                         -1, -1, 0,  -2, -2, -2};

struct PipelineAngularBands {
  int ell_max_first = 0;
  int ell_max_second = 0;
};

inline int pipeline_field_ell_max(const std::size_t field,
                                  const PipelineAngularBands bands) {
  if (field >= point_pipeline_field_count) {
    throw std::out_of_range("pipeline field index is outside the state");
  }
  return field >= static_cast<std::size_t>(PipelineField::SecondP)
             ? bands.ell_max_second
             : bands.ell_max_first;
}

struct PipelineBandResidual {
  double absolute = 0.0;
  double relative = 0.0;
};

struct PipelineBandReport {
  std::array<PipelineBandResidual, point_pipeline_field_count> fields{};
  PipelineBandResidual combined{};
};

namespace pipeline_band_detail {

inline void validate_bands(const ModeRegistry& registry,
                           const PipelineAngularBands bands,
                           const std::size_t theta_count) {
  if (bands.ell_max_first < 2 || bands.ell_max_second < 2) {
    throw std::invalid_argument("pipeline angular bandlimits must be at least 2");
  }
  if (theta_count < static_cast<std::size_t>(
                        std::max(bands.ell_max_first,
                                 bands.ell_max_second) + 1)) {
    throw std::invalid_argument("pipeline angular bands do not fit the theta grid");
  }
  for (const int m : registry.parents()) {
    if (std::abs(m) > bands.ell_max_first) {
      throw std::invalid_argument("parent mode lies outside ell_max_first");
    }
  }
  for (const int m : registry.targets()) {
    if (std::abs(m) > bands.ell_max_second) {
      throw std::invalid_argument("target mode lies outside ell_max_second");
    }
  }
}

template <class StateView>
void validate_state_shape(const StateView& state, const ModeRegistry& registry) {
  static_assert(StateView::rank == 4,
                "pipeline band operations require a rank-four state");
  if (state.extent(0) != registry.size() ||
      state.extent(1) != point_pipeline_field_count ||
      state.extent(2) == 0 || state.extent(3) == 0) {
    throw std::invalid_argument("pipeline state shape does not match registry");
  }
}

inline double magnitude_squared(const Complex value) {
  const double magnitude = Kokkos::abs(value);
  return magnitude * magnitude;
}

inline bool field_mode_is_active(const ModeRegistry& registry, const int mode,
                                 const std::size_t field) {
  return field >= static_cast<std::size_t>(PipelineField::SecondP)
             ? registry.is_target(mode)
             : registry.is_parent(mode);
}

}  // namespace pipeline_band_detail

// Measure the component orthogonal to every declared fixed-(s,m) retained
// band. The report takes a maximum over radial lines, so a localized invalid
// import cannot be hidden by a large global state norm.
template <class StateView>
PipelineBandReport measure_pipeline_state_off_band(
    const StateView& state, const ModeRegistry& registry,
    const PipelineAngularBands bands) {
  pipeline_band_detail::validate_state_shape(state, registry);
  pipeline_band_detail::validate_bands(registry, bands, state.extent(3));
  PipelineBandReport report;
  std::vector<Complex> nodal(state.extent(3));
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t field = 0; field < point_pipeline_field_count; ++field) {
      const int ell_max = pipeline_field_ell_max(field, bands);
      const bool active =
          pipeline_band_detail::field_mode_is_active(registry, m, field);
      std::vector<Complex> projected(state.extent(3), Complex(0.0, 0.0));
      const auto transform = active
          ? std::make_unique<angular::SpinWeightedTransform>(
                pipeline_field_spins[field], m, ell_max,
                static_cast<int>(state.extent(3)))
          : nullptr;
      for (std::size_t radial = 0; radial < state.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < state.extent(3); ++theta) {
          nodal[theta] = state(mode, field, radial, theta);
        }
        if (active) {
          projected = transform->synthesize(transform->analyze(nodal));
        } else {
          std::fill(projected.begin(), projected.end(), Complex(0.0, 0.0));
        }
        double line_squared = 0.0;
        double residual_squared = 0.0;
        double residual_maximum = 0.0;
        for (std::size_t theta = 0; theta < nodal.size(); ++theta) {
          line_squared += pipeline_band_detail::magnitude_squared(nodal[theta]);
          const double residual = Kokkos::abs(nodal[theta] - projected[theta]);
          residual_squared += residual * residual;
          residual_maximum = std::max(residual_maximum, residual);
        }
        const double relative =
            std::sqrt(residual_squared) /
            std::max(std::sqrt(line_squared),
                     std::numeric_limits<double>::min());
        auto& field_report = report.fields[field];
        field_report.absolute =
            std::max(field_report.absolute, residual_maximum);
        field_report.relative = std::max(field_report.relative, relative);
      }
      report.combined.absolute =
          std::max(report.combined.absolute, report.fields[field].absolute);
      report.combined.relative =
          std::max(report.combined.relative, report.fields[field].relative);
    }
  }
  return report;
}

// Analyze/synthesize every field in-place on a host-accessible state. This is
// intended for construction-time finalization, not for restart repair.
template <class StateView>
void project_pipeline_state_to_retained_bands(
    const StateView& state, const ModeRegistry& registry,
    const PipelineAngularBands bands) {
  pipeline_band_detail::validate_state_shape(state, registry);
  pipeline_band_detail::validate_bands(registry, bands, state.extent(3));
  std::vector<Complex> nodal(state.extent(3));
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t field = 0; field < point_pipeline_field_count; ++field) {
      const bool active =
          pipeline_band_detail::field_mode_is_active(registry, m, field);
      const auto transform = active
          ? std::make_unique<angular::SpinWeightedTransform>(
                pipeline_field_spins[field], m,
                pipeline_field_ell_max(field, bands),
                static_cast<int>(state.extent(3)))
          : nullptr;
      for (std::size_t radial = 0; radial < state.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < state.extent(3); ++theta) {
          nodal[theta] = state(mode, field, radial, theta);
        }
        const auto projected = active
            ? transform->synthesize(transform->analyze(nodal))
            : std::vector<Complex>(state.extent(3), Complex(0.0, 0.0));
        for (std::size_t theta = 0; theta < state.extent(3); ++theta) {
          state(mode, field, radial, theta) = projected[theta];
        }
      }
    }
  }
}

inline bool pipeline_state_is_bandlimited(
    const PipelineBandReport& report, const double relative_tolerance) {
  if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
    throw std::invalid_argument("pipeline band tolerance must be nonnegative");
  }
  return std::isfinite(report.combined.relative) &&
         report.combined.relative <= relative_tolerance;
}

// Generic imported-state entry point. Validation finishes before the caller's
// state is mutated. Generated data should instead be explicitly projected by
// its factory before this strict import path is used.
template <class HostStateView>
PipelineBandReport import_pipeline_state(
    const ExecutionSpace& execution, SpatialPipelineStorage& storage,
    const ModeRegistry& registry, const PipelineAngularBands bands,
    const HostStateView& host_state,
    const double relative_tolerance = 5.0e-11) {
  pipeline_band_detail::validate_state_shape(host_state, registry);
  if (host_state.extent(0) != storage.mode_count() ||
      host_state.extent(1) != point_pipeline_field_count ||
      host_state.extent(2) != storage.radial_count() ||
      host_state.extent(3) != storage.theta_count()) {
    throw std::invalid_argument("imported pipeline state shape mismatch");
  }
  const PipelineBandReport report =
      measure_pipeline_state_off_band(host_state, registry, bands);
  if (!pipeline_state_is_bandlimited(report, relative_tolerance)) {
    throw std::runtime_error(
        "imported pipeline state contains meaningful off-band angular content");
  }
  Kokkos::deep_copy(execution, storage.state(), host_state);
  execution.fence("import validated pipeline state");
  return report;
}

}  // namespace teuk
