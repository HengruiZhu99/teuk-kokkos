#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/background.hpp"
#include "teuk/fields.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/reconstruction.hpp"
#include "teuk/sbp.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class ReconstructionAngularInput : std::size_t {
  Eth1F = 0,
  Eth2G = 1,
  Eth2C = 2,
  Eth2Pi = 3,
  EthPrime1BSharp = 4,
  EthPrime2CSharp = 5,
  Count = 6,
};

using ReconstructionRadialStateView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using ReconstructionRadialValueView =
    Kokkos::View<Complex**, Kokkos::LayoutRight, MemorySpace>;
using ReconstructionAngularInputView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using ReconstructionSharpIndexView = Kokkos::View<std::size_t*, MemorySpace>;
using ReconstructionSignedModeView = Kokkos::View<int*, MemorySpace>;

// Fixed-theta signed-mode radial storage. State, RHS, and radial derivative
// ordering is exactly (mode, G/Lambda/H/B/Pi/C/U, radial). The sharp-index view
// is an explicit device lookup for X_m^sharp=conjugate(X_{-m}).
class ReconstructionRadialLines {
 public:
  ReconstructionRadialLines(
      const ModeRegistry& registry, const UniformRadialGrid& grid,
      const std::string& label = "reconstruction_radial_lines")
      : grid_(grid),
        modes_(label + "_modes", registry.size()),
        sharp_indices_(label + "_sharp_indices", registry.size()),
        psi4_(label + "_psi4", registry.size(), grid.size()),
        state_(label + "_state", registry.size(),
               static_cast<std::size_t>(ReconstructionField::Count),
               grid.size()),
        rhs_(label + "_rhs", registry.size(),
             static_cast<std::size_t>(ReconstructionField::Count), grid.size()),
        radial_derivatives_(
            label + "_radial_derivatives", registry.size(),
            static_cast<std::size_t>(ReconstructionField::Count), grid.size()) {
    if (grid.size() < d42_minimum_points) {
      throw std::invalid_argument(
          "SBP reconstruction radial lines require at least eight points");
    }
    if (!registry.is_closed_under_sharp()) {
      throw std::invalid_argument(
          "reconstruction signed modes must be closed under m -> -m");
    }
    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_sharp_indices = Kokkos::create_mirror_view(sharp_indices_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_modes(mode) = registry.modes()[mode];
      host_sharp_indices(mode) =
          registry.sharp_index(registry.modes()[mode]);
    }
    Kokkos::deep_copy(modes_, host_modes);
    Kokkos::deep_copy(sharp_indices_, host_sharp_indices);
  }

  [[nodiscard]] std::size_t mode_count() const { return modes_.extent(0); }
  [[nodiscard]] std::size_t radial_point_count() const { return grid_.size(); }
  [[nodiscard]] const UniformRadialGrid& grid() const { return grid_; }

  [[nodiscard]] ReconstructionSignedModeView modes() const { return modes_; }
  [[nodiscard]] ReconstructionSharpIndexView sharp_indices() const {
    return sharp_indices_;
  }
  [[nodiscard]] ReconstructionRadialValueView psi4() const { return psi4_; }
  [[nodiscard]] ReconstructionRadialStateView state() const { return state_; }
  [[nodiscard]] ReconstructionRadialStateView rhs() const { return rhs_; }
  [[nodiscard]] ReconstructionRadialStateView radial_derivatives() const {
    return radial_derivatives_;
  }

 private:
  UniformRadialGrid grid_;
  ReconstructionSignedModeView modes_;
  ReconstructionSharpIndexView sharp_indices_;
  ReconstructionRadialValueView psi4_;
  ReconstructionRadialStateView state_;
  ReconstructionRadialStateView rhs_;
  ReconstructionRadialStateView radial_derivatives_;
};

// Evaluate all seven reconstruction time derivatives at one fixed theta.
// Angular inputs are stage-local values prepared by the angular subsystem.
// radial_derivatives() is retained after the call and can be passed directly
// to reconstruction residual diagnostics; callers that do not need it may
// simply ignore the view.
inline void evaluate_sbp_reconstruction_radial_lines_rhs(
    ReconstructionRadialLines& lines, const KerrParameters& parameters,
    const double theta, const ReconstructionAngularInputView& angular) {
  if (!(parameters.compactification_length > 0.0)) {
    throw std::invalid_argument("compactification length must be positive");
  }
  const std::size_t mode_count = lines.mode_count();
  const std::size_t point_count = lines.radial_point_count();
  if (angular.extent(0) != mode_count ||
      angular.extent(1) !=
          static_cast<std::size_t>(ReconstructionAngularInput::Count) ||
      angular.extent(2) != point_count) {
    throw std::invalid_argument(
        "reconstruction angular input extents do not match radial lines");
  }

  const UniformRadialGrid grid = lines.grid();
  const double inverse_spacing = 1.0 / grid.spacing();
  const std::size_t field_count =
      static_cast<std::size_t>(ReconstructionField::Count);
  const auto state = lines.state();
  const auto radial_derivatives = lines.radial_derivatives();
  const auto psi4 = lines.psi4();
  const auto rhs = lines.rhs();
  const auto sharp_indices = lines.sharp_indices();

  ExecutionSpace execution;
  const std::size_t derivative_points = mode_count * field_count * point_count;
  const Kokkos::RangePolicy<ExecutionSpace> derivative_policy(
      execution, 0,
      static_cast<typename ExecutionSpace::size_type>(derivative_points));
  Kokkos::parallel_for(
      "teuk_sbp_reconstruction_radial_derivatives", derivative_policy,
      KOKKOS_LAMBDA(const std::size_t flat_index) {
        const std::size_t mode_field = flat_index / point_count;
        const std::size_t radial = flat_index - mode_field * point_count;
        const std::size_t mode = mode_field / field_count;
        const std::size_t field = mode_field - mode * field_count;
        radial_derivatives(mode, field, radial) = d42_first_derivative_at(
            &state(mode, field, 0), point_count, radial, inverse_spacing);
      });

  const std::size_t total_points = mode_count * point_count;
  const Kokkos::RangePolicy<ExecutionSpace> rhs_policy(
      execution, 0,
      static_cast<typename ExecutionSpace::size_type>(total_points));
  Kokkos::parallel_for(
      "teuk_sbp_reconstruction_rhs", rhs_policy,
      KOKKOS_LAMBDA(const std::size_t flat_index) {
        const std::size_t mode = flat_index / point_count;
        const std::size_t radial = flat_index - mode * point_count;
        const std::size_t sharp_mode = sharp_indices(mode);
        const double radius = grid.coordinate(radial);
        const KerrBackgroundPoint background =
            kerr_background_point(parameters, radius, theta);

        constexpr std::size_t G =
            static_cast<std::size_t>(ReconstructionField::G);
        constexpr std::size_t Lambda =
            static_cast<std::size_t>(ReconstructionField::Lambda);
        constexpr std::size_t H =
            static_cast<std::size_t>(ReconstructionField::H);
        constexpr std::size_t B =
            static_cast<std::size_t>(ReconstructionField::B);
        constexpr std::size_t Pi =
            static_cast<std::size_t>(ReconstructionField::Pi);
        constexpr std::size_t C =
            static_cast<std::size_t>(ReconstructionField::C);
        constexpr std::size_t U =
            static_cast<std::size_t>(ReconstructionField::U);

        const ReconstructionFields point_fields{
            psi4(mode, radial),
            state(mode, G, radial),
            state(mode, H, radial),
            state(mode, Lambda, radial),
            state(mode, Pi, radial),
            state(mode, B, radial),
            state(mode, C, radial),
            state(mode, U, radial),
            Kokkos::conj(state(sharp_mode, Pi, radial)),
            Kokkos::conj(state(sharp_mode, B, radial)),
            Kokkos::conj(state(sharp_mode, C, radial))};
        const ReconstructionAngularDerivatives point_angular{
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::Eth1F),
                    radial),
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::Eth2G),
                    radial),
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::Eth2C),
                    radial),
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::Eth2Pi),
                    radial),
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::EthPrime1BSharp),
                    radial),
            angular(mode,
                    static_cast<std::size_t>(
                        ReconstructionAngularInput::EthPrime2CSharp),
                    radial)};
        const ReconstructionDeltaRhs delta_rhs =
            reconstruction_delta_rhs(radius, background, point_fields,
                                     point_angular);

        // Preserve the transport dependency order G,Lambda,H,B,Pi,C,U.
        rhs(mode, G, radial) = reconstruction_time_derivative(
            point_fields.G, radial_derivatives(mode, G, radial), delta_rhs.G, 2,
            radius, parameters.mass, parameters.compactification_length);
        rhs(mode, Lambda, radial) = reconstruction_time_derivative(
            point_fields.Lambda, radial_derivatives(mode, Lambda, radial),
            delta_rhs.Lambda, 1, radius, parameters.mass,
            parameters.compactification_length);
        rhs(mode, H, radial) = reconstruction_time_derivative(
            point_fields.H, radial_derivatives(mode, H, radial), delta_rhs.H, 3,
            radius, parameters.mass, parameters.compactification_length);
        rhs(mode, B, radial) = reconstruction_time_derivative(
            point_fields.B, radial_derivatives(mode, B, radial), delta_rhs.B, 1,
            radius, parameters.mass, parameters.compactification_length);
        rhs(mode, Pi, radial) = reconstruction_time_derivative(
            point_fields.Pi, radial_derivatives(mode, Pi, radial), delta_rhs.Pi,
            2, radius, parameters.mass, parameters.compactification_length);
        rhs(mode, C, radial) = reconstruction_time_derivative(
            point_fields.C, radial_derivatives(mode, C, radial), delta_rhs.C, 2,
            radius, parameters.mass, parameters.compactification_length);
        rhs(mode, U, radial) = reconstruction_time_derivative(
            point_fields.U, radial_derivatives(mode, U, radial), delta_rhs.U, 3,
            radius, parameters.mass, parameters.compactification_length);
      });
}

}  // namespace teuk
