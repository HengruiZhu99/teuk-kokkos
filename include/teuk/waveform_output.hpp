#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/spatial_pipeline.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class WaveformBoundary { Scri, Horizon };

struct EndpointWaveformRecord {
  std::uint64_t step = 0;
  double time = 0.0;
  WaveformBoundary boundary = WaveformBoundary::Scri;
  int perturbative_order = 1;
  int m = 0;
  int ell = 2;
  Complex rescaled = Complex(0.0, 0.0);
  Complex endpoint_waveform = Complex(0.0, 0.0);
};

// Extract fixed-m spin--2 modal coefficients without copying the full state.
// At scri, endpoint_waveform=L^2 F=lim_{r->infinity} r Psi4.  At the horizon,
// endpoint_waveform=R_H F=Psi4 in the repository's rotated Kinnersley tetrad.
// The horizon value is therefore convention-fixed rather than a regular-
// tetrad flux observable; the rescaled F coefficient is retained explicitly.
class EndpointWaveformSampler {
 public:
  EndpointWaveformSampler(const ModeRegistry& registry,
                          const PipelineAngularBands bands,
                          const int theta_nodes,
                          const double compactification_length,
                          const double horizon_radius)
      : registry_(&registry),
        theta_nodes_(theta_nodes),
        compactification_length_squared_(compactification_length *
                                         compactification_length),
        horizon_radius_(horizon_radius),
        device_endpoint_("endpoint_waveform_device", registry.size(),
                         static_cast<std::size_t>(theta_nodes)),
        host_endpoint_("endpoint_waveform_host", registry.size(),
                       static_cast<std::size_t>(theta_nodes)) {
    if (theta_nodes <= 0 || !(compactification_length > 0.0) ||
        !(horizon_radius > 0.0)) {
      throw std::invalid_argument("endpoint waveform geometry is invalid");
    }
    first_.resize(registry.size());
    second_.resize(registry.size());
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      const int m = registry.modes()[mode];
      if (registry.is_parent(m)) {
        first_[mode] = std::make_unique<angular::SpinWeightedTransform>(
            -2, m, bands.ell_max_first, theta_nodes);
      }
      if (registry.is_target(m)) {
        second_[mode] = std::make_unique<angular::SpinWeightedTransform>(
            -2, m, bands.ell_max_second, theta_nodes);
      }
    }
  }

  template <class Execution, class StateView>
  std::vector<EndpointWaveformRecord> sample(
      const Execution& execution, const StateView& state,
      const std::uint64_t step, const double time) {
    static_assert(StateView::rank == 4,
                  "endpoint waveform input must have rank four");
    if (state.extent(0) != registry_->size() ||
        state.extent(1) != point_pipeline_field_count ||
        state.extent(2) == 0 ||
        state.extent(3) != static_cast<std::size_t>(theta_nodes_)) {
      throw std::invalid_argument("endpoint waveform state extents mismatch");
    }
    std::vector<EndpointWaveformRecord> records;
    const std::size_t first_field =
        static_cast<std::size_t>(PipelineField::FirstPsi);
    const std::size_t second_field =
        static_cast<std::size_t>(PipelineField::SecondPsi);
    sample_field(execution, state, first_field, 1, first_, step, time,
                 records);
    sample_field(execution, state, second_field, 2, second_, step, time,
                 records);
    return records;
  }

 private:
  template <class Execution, class StateView>
  void sample_field(
      const Execution& execution, const StateView& state,
      const std::size_t field, const int perturbative_order,
      const std::vector<std::unique_ptr<angular::SpinWeightedTransform>>&
          transforms,
      const std::uint64_t step, const double time,
      std::vector<EndpointWaveformRecord>& records) {
    std::vector<Complex> nodal(static_cast<std::size_t>(theta_nodes_));
    for (const WaveformBoundary boundary :
         {WaveformBoundary::Scri, WaveformBoundary::Horizon}) {
      const std::size_t radial =
          boundary == WaveformBoundary::Scri ? 0 : state.extent(2) - 1;
      const auto device_endpoint = device_endpoint_;
      const std::size_t theta_count = static_cast<std::size_t>(theta_nodes_);
      const std::size_t total = registry_->size() * theta_count;
      Kokkos::parallel_for(
          "gather_endpoint_waveform_slice",
          Kokkos::RangePolicy<Execution>(execution, 0, total),
          KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t mode = flat / theta_count;
            const std::size_t theta = flat - mode * theta_count;
            device_endpoint(mode, theta) = state(mode, field, radial, theta);
          });
      Kokkos::deep_copy(execution, host_endpoint_, device_endpoint_);
      execution.fence("copy endpoint waveform slice");
      const double endpoint_scale =
          boundary == WaveformBoundary::Scri
              ? compactification_length_squared_
              : horizon_radius_;
      for (std::size_t mode = 0; mode < registry_->size(); ++mode) {
        if (!transforms[mode]) continue;
        for (std::size_t theta = 0; theta < nodal.size(); ++theta) {
          nodal[theta] = host_endpoint_(mode, theta);
        }
        const auto modal = transforms[mode]->analyze(nodal);
        for (std::size_t index = 0; index < modal.size(); ++index) {
          records.push_back(
              {step,
               time,
               boundary,
               perturbative_order,
               registry_->modes()[mode],
               transforms[mode]->ell_min() + static_cast<int>(index),
               modal[index],
               endpoint_scale * modal[index]});
        }
      }
    }
  }

  const ModeRegistry* registry_;
  int theta_nodes_;
  double compactification_length_squared_;
  double horizon_radius_;
  Kokkos::View<Complex**, Kokkos::LayoutRight, MemorySpace> device_endpoint_;
  Kokkos::View<Complex**, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_endpoint_;
  std::vector<std::unique_ptr<angular::SpinWeightedTransform>> first_;
  std::vector<std::unique_ptr<angular::SpinWeightedTransform>> second_;
};

inline const char* waveform_boundary_name(const WaveformBoundary boundary) {
  return boundary == WaveformBoundary::Scri ? "scri" : "horizon";
}

inline void write_endpoint_waveform_header(std::ostream& output) {
  output << "step,time,boundary,order,m,ell,rescaled_re,rescaled_im,"
            "endpoint_re,endpoint_im\n";
}

inline void write_endpoint_waveform_records(
    std::ostream& output,
    const std::vector<EndpointWaveformRecord>& records) {
  output << std::setprecision(17);
  for (const auto& record : records) {
    output << record.step << ',' << record.time << ','
           << waveform_boundary_name(record.boundary) << ','
           << record.perturbative_order << ',' << record.m << ','
           << record.ell << ',' << record.rescaled.real() << ','
           << record.rescaled.imag() << ','
           << record.endpoint_waveform.real() << ','
           << record.endpoint_waveform.imag() << '\n';
  }
}

}  // namespace teuk
