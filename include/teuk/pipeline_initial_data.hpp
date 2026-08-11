#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/background.hpp"
#include "teuk/fields.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/sbp.hpp"
#include "teuk/spatial_pipeline.hpp"
#include "teuk/teukolsky.hpp"

namespace teuk {

struct GaussianPulseMode {
  int ell;
  int m;
  Complex amplitude;
};

struct PipelineGaussianPulse {
  // The stored first-order field is
  //   Psi(R,theta) = exp(-((R-center)/width)^2)
  //                  sum_l amplitude_lm {}_{-2}Y_lm(theta).
  Real center = 0.4;
  Real width = 0.1;
  std::vector<GaussianPulseMode> modes;

  // Initial coordinate-time derivative is exactly zero. Q is computed by the
  // same D4-2 operator used by the production radial path, and P is then set to
  // -2 K Q + G_m Psi so teukolsky_psi_rhs returns zero pointwise.

  // Default nonlinear/reconstruction data are explicit zeros. Nonzero scales
  // are diagnostic opt-ins: second order copies the complete consistent
  // first-order (P,Q,Psi) state, while each reconstruction field receives the
  // same radial/mode seed synthesized at that field's correct spin weight.
  Complex second_order_scale = Complex(0.0, 0.0);
  std::array<Complex,
             static_cast<std::size_t>(ReconstructionField::Count)>
      reconstruction_scales{};
};

namespace pipeline_initial_data_detail {

inline bool is_zero(const Complex value) {
  return value.real() == 0.0 && value.imag() == 0.0;
}

inline void validate_gaussian_pulse(
    const SpatialPipelineStorage& storage, const ModeRegistry& registry,
    const int ell_max, const KerrParameters& background,
    const PipelineGaussianPulse& pulse) {
  if (storage.mode_count() != registry.size()) {
    throw std::invalid_argument(
        "initial-data registry does not match pipeline mode extent");
  }
  if (!registry.is_closed_under_sharp()) {
    throw std::invalid_argument(
        "initial-data modes must be closed under m -> -m");
  }
  if (ell_max < 2 ||
      storage.theta_count() < static_cast<std::size_t>(ell_max + 1)) {
    throw std::invalid_argument(
        "initial-data angular band does not fit the pipeline grid");
  }
  if (storage.radial_count() < d42_minimum_points) {
    throw std::invalid_argument(
        "Gaussian reduction data require a D4-2 radial grid");
  }
  if (!(pulse.width > 0.0) ||
      pulse.center < storage.radial_grid().lower_radius() ||
      pulse.center > storage.radial_grid().upper_radius()) {
    throw std::invalid_argument(
        "Gaussian center must be on-grid and width must be positive");
  }
  if (!(background.mass > 0.0) ||
      std::abs(background.spin) > background.mass ||
      !(background.compactification_length > 0.0)) {
    throw std::invalid_argument("initial-data Kerr parameters are invalid");
  }
  if (pulse.modes.empty()) {
    throw std::invalid_argument("Gaussian pulse needs at least one mode");
  }
  for (std::size_t i = 0; i < pulse.modes.size(); ++i) {
    const auto& seed = pulse.modes[i];
    if (!registry.contains(seed.m)) {
      throw std::invalid_argument("Gaussian seed m is not registered");
    }
    if (seed.ell < std::max(2, std::abs(seed.m)) || seed.ell > ell_max) {
      throw std::invalid_argument("Gaussian seed ell is outside the band");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (pulse.modes[j].ell == seed.ell &&
          pulse.modes[j].m == seed.m) {
        throw std::invalid_argument("duplicate Gaussian (ell,m) seed");
      }
    }
  }
}

inline std::vector<Complex> angular_seed(
    const PipelineGaussianPulse& pulse, const int spin, const int m,
    const int ell_max, const int theta_nodes) {
  const angular::SpinWeightedTransform transform(spin, m, ell_max,
                                                 theta_nodes);
  std::vector<Complex> modal(transform.mode_count(), Complex(0.0, 0.0));
  for (const auto& seed : pulse.modes) {
    if (seed.m == m) {
      modal[static_cast<std::size_t>(seed.ell - transform.ell_min())] +=
          seed.amplitude;
    }
  }
  return transform.synthesize(modal);
}

}  // namespace pipeline_initial_data_detail

// Readable host setup for the complete SpatialPipeline state. All thirteen
// fields are written explicitly into one host mirror, followed by exactly one
// host-to-caller deep copy and a setup fence. No production timestep uses this
// allocation-owning routine.
inline void initialize_compactified_gaussian_pulse(
    const ExecutionSpace& execution, SpatialPipeline& pipeline,
    const ModeRegistry& registry, const int ell_max,
    const KerrParameters& background, const PipelineGaussianPulse& pulse) {
  auto& storage = pipeline.storage();
  pipeline_initial_data_detail::validate_gaussian_pulse(
      storage, registry, ell_max, background, pulse);
  const auto& radial_grid = storage.radial_grid();
  const std::size_t radial_count = storage.radial_count();
  const std::size_t theta_count = storage.theta_count();

  Kokkos::View<Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_state("pipeline_gaussian_host_state", registry.size(),
                 point_pipeline_field_count, radial_count, theta_count);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < point_pipeline_field_count; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          host_state(mode, field, radial, theta) = Complex(0.0, 0.0);
        }
      }
    }
  }

  const auto first_p = static_cast<std::size_t>(PipelineField::FirstP);
  const auto first_q = static_cast<std::size_t>(PipelineField::FirstQ);
  const auto first_psi = static_cast<std::size_t>(PipelineField::FirstPsi);
  const auto second_p = static_cast<std::size_t>(PipelineField::SecondP);
  const auto second_q = static_cast<std::size_t>(PipelineField::SecondQ);
  const auto second_psi = static_cast<std::size_t>(PipelineField::SecondPsi);

  // Synthesize the first-order angular seed and apply the common compactified
  // radial Gaussian.
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    const auto angular_values = pipeline_initial_data_detail::angular_seed(
        pulse, -2, m, ell_max, static_cast<int>(theta_count));
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const Real coordinate = radial_grid.coordinate(radial);
      const Real normalized = (coordinate - pulse.center) / pulse.width;
      const Real profile = std::exp(-normalized * normalized);
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        host_state(mode, first_psi, radial, theta) =
            profile * angular_values[theta];
      }
    }
  }

  // Use the exact host D4-2 reference on every (m,theta) radial line.
  std::vector<Complex> psi_line(radial_count);
  std::vector<Complex> q_line(radial_count);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        psi_line[radial] = host_state(mode, first_psi, radial, theta);
      }
      d42_first_derivative(radial_grid, psi_line, q_line);
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        host_state(mode, first_q, radial, theta) = q_line[radial];
      }
    }
  }

  // P is defined from the chosen partial_T Psi=0 condition.
  const auto angular_grid =
      angular::gauss_legendre(static_cast<int>(theta_count));
  TeukolskyParameters teukolsky;
  teukolsky.mass = background.mass;
  teukolsky.spin = background.spin;
  teukolsky.compactification_length = background.compactification_length;
  teukolsky.spin_weight = -2;
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    teukolsky.azimuthal_mode = registry.modes()[mode];
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const Real coordinate = radial_grid.coordinate(radial);
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        const auto coefficients = teukolsky_coefficients(
            teukolsky, coordinate, angular_grid.theta(theta));
        const Complex psi = host_state(mode, first_psi, radial, theta);
        const Complex q = host_state(mode, first_q, radial, theta);
        host_state(mode, first_p, radial, theta) =
            -2.0 * coefficients.radial_advection * q +
            coefficients.definition * psi;
      }
    }
  }

  if (!pipeline_initial_data_detail::is_zero(pulse.second_order_scale)) {
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          host_state(mode, second_p, radial, theta) =
              pulse.second_order_scale *
              host_state(mode, first_p, radial, theta);
          host_state(mode, second_q, radial, theta) =
              pulse.second_order_scale *
              host_state(mode, first_q, radial, theta);
          host_state(mode, second_psi, radial, theta) =
              pulse.second_order_scale *
              host_state(mode, first_psi, radial, theta);
        }
      }
    }
  }

  constexpr std::array<PipelineField, 7> reconstruction_fields{
      PipelineField::G, PipelineField::Lambda, PipelineField::H,
      PipelineField::B, PipelineField::Pi, PipelineField::C, PipelineField::U};
  for (std::size_t field = 0; field < reconstruction_fields.size(); ++field) {
    const Complex scale = pulse.reconstruction_scales[field];
    if (pipeline_initial_data_detail::is_zero(scale)) continue;
    const int field_spin = reconstruction_metadata[field].spin;
    const std::size_t pipeline_field =
        static_cast<std::size_t>(reconstruction_fields[field]);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      const auto angular_values = pipeline_initial_data_detail::angular_seed(
          pulse, field_spin, registry.modes()[mode], ell_max,
          static_cast<int>(theta_count));
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        const Real coordinate = radial_grid.coordinate(radial);
        const Real normalized = (coordinate - pulse.center) / pulse.width;
        const Real profile = std::exp(-normalized * normalized);
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          host_state(mode, pipeline_field, radial, theta) =
              scale * profile * angular_values[theta];
        }
      }
    }
  }

  Kokkos::deep_copy(execution, storage.state(), host_state);
  execution.fence("initialize compactified Gaussian pipeline pulse");
}

}  // namespace teuk
