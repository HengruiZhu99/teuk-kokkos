#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/angular.hpp"
#include "teuk/angular_coordinator.hpp"
#include "teuk/fields.hpp"
#include "teuk/modes.hpp"
#include "teuk/routeb_reconstruction_jet.hpp"
#include "teuk/routeb_teukolsky_primary_jet.hpp"

namespace teuk {

namespace routeb_angular_jet_detail {

using Stride4 = std::array<std::size_t, 4>;

KOKKOS_INLINE_FUNCTION std::size_t index4(
    const std::size_t a, const std::size_t b, const std::size_t c,
    const std::size_t d, const Stride4& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3];
}

struct ApplyGhpJetFunctor {
  const Complex* field;
  const Complex* dt_field;
  const Complex* pure_angular;
  Complex* output;
  const Real* radius_values;
  Stride4 field_stride;
  Stride4 dt_stride;
  Stride4 angular_stride;
  Stride4 output_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t active_orders;
  int spin;
  int boost;
  Real kerr_spin;
  Real length;
  const Real* sin_theta;
  const Real* cos_theta;
  bool prime;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    using Jet = RouteBRadialTaylorJet<4, Complex>;
    Jet value;
    Jet dt;
    Jet pure;
    for (std::size_t order = 0; order < active_orders; ++order) {
      value[order] = field[index4(mode, order, radial, theta, field_stride)];
      dt[order] =
          dt_field[index4(mode, order, radial, theta, dt_stride)];
      pure[order] = pure_angular[index4(mode, order, radial, theta,
                                         angular_stride)];
    }
    Jet R = Jet::constant(Complex(0.0, 0.0));
    const Real radius0 = radius_values[radial];
    R[0] = Complex(radius0, 0.0);
    R[1] = Complex(1.0, 0.0);
    const Real length2 = length * length;
    const Real sine = sin_theta[theta];
    const Real cosine = cos_theta[theta];
    const Complex imaginary(0.0, 1.0);
    const Complex sign = prime ? Complex(0.0, kerr_spin * cosine)
                               : Complex(0.0, -kerr_spin * cosine);
    const Jet denominator = Jet::constant(Complex(length2, 0.0)) + sign * R;
    const Complex dt_factor =
        prime ? Complex(0.0, kerr_spin * sine)
              : Complex(0.0, -kerr_spin * sine);
    const int weight = prime ? -spin + boost : spin + boost;
    const Complex connection_factor =
        prime ? Complex(0.0, static_cast<Real>(weight) * kerr_spin * sine)
              : Complex(0.0, -static_cast<Real>(weight) * kerr_spin * sine);
    const Real inverse_sqrt_two = 1.0 / Kokkos::sqrt(2.0);
    const Jet result =
        Complex(inverse_sqrt_two, 0.0) * (dt_factor * dt + pure) /
            denominator +
        Complex(inverse_sqrt_two, 0.0) * connection_factor * R * value /
            (denominator * denominator);
    for (std::size_t order = 0; order < 4; ++order) {
      output[index4(mode, order, radial, theta, output_stride)] =
          order < active_orders ? result[order] : Complex{};
    }
  }

};

struct MarkRank4Rows {
  Complex* output;
  std::uint64_t* output_stamps;
  const std::uint64_t* source_a;
  const std::uint64_t* source_b;
  Stride4 output_stride;
  Stride4 output_stamp_stride;
  Stride4 source_a_stride;
  Stride4 source_b_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t first_order;
  std::size_t active_orders;
  std::uint64_t source_a_token;
  std::uint64_t source_b_token;
  std::uint64_t output_token;
  bool use_source_b;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t rows_per_mode = active_orders * radial_count;
    const std::size_t mode = flat / rows_per_mode;
    const std::size_t within = flat - mode * rows_per_mode;
    const std::size_t local_order = within / radial_count;
    const std::size_t order = first_order + local_order;
    const std::size_t radial = within - local_order * radial_count;
    bool valid = true;
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const Complex value =
          output[index4(mode, order, radial, theta, output_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag()) &&
              source_a[index4(mode, order, radial, theta,
                              source_a_stride)] == source_a_token &&
              (!use_source_b ||
               source_b[index4(mode, order, radial, theta,
                               source_b_stride)] == source_b_token);
    }
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const auto index = index4(mode, order, radial, theta, output_stride);
      if (!valid) output[index] = {};
      output_stamps[index4(mode, order, radial, theta,
                           output_stamp_stride)] = valid ? output_token : 0;
    }
  }
};

using Stride5 = std::array<std::size_t, 5>;
KOKKOS_INLINE_FUNCTION std::size_t index5(
    const std::size_t a, const std::size_t b, const std::size_t c,
    const std::size_t d, const std::size_t e, const Stride5& stride) {
  return a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3] +
         e * stride[4];
}

struct MarkRank5Rows {
  Complex* output;
  std::uint64_t* output_stamps;
  const std::uint64_t* source_stamps;
  Stride5 output_stride;
  Stride5 output_stamp_stride;
  Stride5 source_stride;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t first_field;
  std::size_t field_count;
  std::size_t active_orders;
  std::uint64_t source_token;
  std::uint64_t output_token;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t rows_per_mode = field_count * active_orders * radial_count;
    const std::size_t mode = flat / rows_per_mode;
    const std::size_t within_mode = flat - mode * rows_per_mode;
    const std::size_t field_local =
        within_mode / (active_orders * radial_count);
    const std::size_t within_field =
        within_mode - field_local * active_orders * radial_count;
    const std::size_t order = within_field / radial_count;
    const std::size_t radial = within_field - order * radial_count;
    const std::size_t field = first_field + field_local;
    bool valid = true;
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const Complex value =
          output[index5(mode, field, order, radial, theta, output_stride)];
      valid = valid && Kokkos::isfinite(value.real()) &&
              Kokkos::isfinite(value.imag()) &&
              source_stamps[index5(mode, field, order, radial, theta,
                                    source_stride)] == source_token;
    }
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const auto index = index5(mode, field, order, radial, theta,
                                output_stride);
      if (!valid) output[index] = {};
      output_stamps[index5(mode, field, order, radial, theta,
                           output_stamp_stride)] = valid ? output_token : 0;
    }
  }
};

static_assert(std::is_trivially_copyable_v<ApplyGhpJetFunctor>);
static_assert(std::is_trivially_copyable_v<MarkRank4Rows>);
static_assert(std::is_trivially_copyable_v<MarkRank5Rows>);

}  // namespace routeb_angular_jet_detail

// Closed, source-independent Route-B linear jet graph. The coordinator owns
// both approved radial towers and interleaves a componentwise angular action
// and Galerkin projection at every normalized radial coefficient. All methods
// after construction enqueue work on one ordered execution-space instance;
// they allocate nothing and fence nowhere.
template <class ExecutionSpace = teuk::ExecutionSpace>
class RouteBAngularJetCoordinator {
 public:
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using complex4 =
      Kokkos::View<Complex****, Kokkos::LayoutRight, memory_space>;
  using stamp4 =
      Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, memory_space>;
  using complex5 =
      Kokkos::View<Complex*****, Kokkos::LayoutRight, memory_space>;
  using stamp5 =
      Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, memory_space>;

  RouteBAngularJetCoordinator(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& grid, const int ell_max, const int node_count,
      const TeukolskyParameters& parameters,
      const std::string& label = "routeb_angular_jet")
      : registry_(registry),
        grid_(grid),
        ell_max_(ell_max),
        node_count_(checked_node_count(registry, grid, ell_max, node_count,
                                       parameters)),
        parameters_(parameters),
        background_{parameters.mass, parameters.spin,
                    parameters.compactification_length},
        modes_(label + "_modes", registry.size()),
        sharp_(label + "_sharp", registry.size()),
        theta_(label + "_theta", node_count),
        radius_(label + "_radius", grid.size()),
        sin_theta_(label + "_sin_theta", node_count),
        cos_theta_(label + "_cos_theta", node_count),
        primary_(registry.size(), grid, node_count, label + "_primary"),
        reconstruction_(registry.size(), grid, node_count,
                        label + "_reconstruction"),
        first_angular_(execution, registry, -2, -2, ell_max, node_count,
                       grid.size(), background_),
        g_angular_(execution, registry, -1, -1, ell_max, node_count,
                   grid.size(), background_),
        pi_angular_(execution, registry, -1, 0, ell_max, node_count,
                    grid.size(), background_),
        c_angular_(execution, registry, -1, 1, ell_max, node_count,
                   grid.size(), background_),
        scalar_angular_(execution, registry, 0, 0, ell_max, node_count,
                        grid.size(), background_),
        b_sharp_angular_(execution, registry, 2, 0, ell_max, node_count,
                         grid.size(), background_),
        c_sharp_angular_(execution, registry, 1, 1, ell_max, node_count,
                         grid.size(), background_),
        primary_snapshot_(label + "_primary_snapshot", registry.size(), 4,
                          grid.size(), node_count),
        primary_snapshot_stamps_(label + "_primary_snapshot_stamps",
                                 registry.size(), 4, grid.size(), node_count),
        primary_laplacian_(label + "_primary_laplacian", registry.size(), 4,
                           grid.size(), node_count),
        primary_laplacian_stamps_(label + "_primary_laplacian_stamps",
                                  registry.size(), 4, grid.size(), node_count),
        primary_projected_(label + "_primary_projected", registry.size(), 3,
                           5, grid.size(), node_count),
        primary_projected_stamps_(label + "_primary_projected_stamps",
                                  registry.size(), 3, 5, grid.size(),
                                  node_count),
        reconstruction_angular_(label + "_reconstruction_angular",
                                registry.size(), 4, grid.size(), node_count),
        reconstruction_pure_(label + "_reconstruction_pure", registry.size(),
                             4, grid.size(), node_count),
        reconstruction_angular_stamps_(
            label + "_reconstruction_angular_stamps", registry.size(), 4,
            grid.size(), node_count),
        reconstruction_pass3_(label + "_reconstruction_pass3",
                              registry.size(), 4, 4, grid.size(), node_count),
        reconstruction_pass3_stamps_(
            label + "_reconstruction_pass3_stamps", registry.size(), 4, 4,
            grid.size(), node_count),
        reconstruction_projected_(label + "_reconstruction_projected",
                                  registry.size(), 7, 5, grid.size(),
                                  node_count),
        reconstruction_projected_stamps_(
            label + "_reconstruction_projected_stamps", registry.size(), 7,
            5, grid.size(), node_count),
        sharp_pairs_(label + "_sharp_pairs", registry.size(), 4, 4,
                     grid.size(), node_count),
        sharp_pair_stamps_(label + "_sharp_pair_stamps", registry.size(), 4,
                           4, grid.size(), node_count) {
    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_sharp = Kokkos::create_mirror_view(sharp_);
    auto host_theta = Kokkos::create_mirror_view(theta_);
    auto host_radius = Kokkos::create_mirror_view(radius_);
    auto host_sin = Kokkos::create_mirror_view(sin_theta_);
    auto host_cos = Kokkos::create_mirror_view(cos_theta_);
    const auto angular_grid = angular::gauss_legendre(node_count_);
    for (std::size_t mode = 0; mode < registry_.size(); ++mode) {
      host_modes(mode) = registry_.modes()[mode];
      host_sharp(mode) = registry_.sharp_index(registry_.modes()[mode]);
    }
    for (int node = 0; node < node_count_; ++node) {
      const std::size_t index = static_cast<std::size_t>(node);
      host_theta(index) = angular_grid.theta(index);
      host_sin(index) = std::sin(host_theta(index));
      host_cos(index) = std::cos(host_theta(index));
    }
    for (std::size_t radial = 0; radial < grid_.size(); ++radial) {
      host_radius(radial) = grid_.coordinate(radial);
    }
    Kokkos::deep_copy(execution, modes_, host_modes);
    Kokkos::deep_copy(execution, sharp_, host_sharp);
    Kokkos::deep_copy(execution, theta_, host_theta);
    Kokkos::deep_copy(execution, radius_, host_radius);
    Kokkos::deep_copy(execution, sin_theta_, host_sin);
    Kokkos::deep_copy(execution, cos_theta_, host_cos);
    execution.fence("initialize Route-B angular jet coordinates");
  }

  RouteBAngularJetCoordinator(const RouteBAngularJetCoordinator&) = delete;
  RouteBAngularJetCoordinator& operator=(const RouteBAngularJetCoordinator&) =
      delete;
  RouteBAngularJetCoordinator(RouteBAngularJetCoordinator&&) = delete;
  RouteBAngularJetCoordinator& operator=(RouteBAngularJetCoordinator&&) =
      delete;

  [[nodiscard]] const auto primary_values() const { return primary_.values(); }
  [[nodiscard]] const auto primary_stamps() const { return primary_.stamps(); }
  [[nodiscard]] const auto reconstruction_values() const {
    return reconstruction_.values();
  }
  [[nodiscard]] const auto reconstruction_stamps() const {
    return reconstruction_.stamps();
  }
  [[nodiscard]] const auto primary_current_coefficients() const {
    return primary_.current_coefficients();
  }
  [[nodiscard]] const auto primary_current_coefficient_stamps() const {
    return primary_.current_coefficient_stamps();
  }
  [[nodiscard]] const auto reconstruction_current_coefficients() const {
    return reconstruction_.current_coefficients();
  }
  [[nodiscard]] const auto reconstruction_current_coefficient_stamps() const {
    return reconstruction_.current_coefficient_stamps();
  }
  [[nodiscard]] const auto theta() const {
    return typename decltype(theta_)::const_type(theta_);
  }
  [[nodiscard]] std::size_t current_level() const {
    return primary_.current_level();
  }
  [[nodiscard]] std::uint64_t generation() const { return generation_; }
  [[nodiscard]] int ell_max() const { return ell_max_; }
  [[nodiscard]] int node_count() const { return node_count_; }

  template <class PrimaryInput, class PrimaryStampView,
            class ReconstructionInput, class ReconstructionStampView>
  void initialize(const execution_space& execution,
                  const PrimaryInput& primary_input,
                  const PrimaryStampView& primary_stamps,
                  const ReconstructionInput& reconstruction_input,
                  const ReconstructionStampView& reconstruction_stamps,
                  const std::uint64_t generation) {
    prevalidate_initial(execution, primary_input, primary_stamps,
                        reconstruction_input, reconstruction_stamps,
                        generation);
    primary_.initialize(execution, parameters_, modes_, theta_, primary_input,
                        primary_stamps, generation,
                        ReductionEvolution::FreeDamped, 0.0);
    reconstruction_.initialize(
        execution, background_, modes_, sharp_, theta_, reconstruction_input,
        reconstruction_stamps, generation, ReductionEvolution::FreeDamped,
        0.0);
    generation_ = generation;
    initialized_ = true;
    project_primary_current(execution);
    project_reconstruction_range(execution,
                                 reconstruction_.current_coefficients(), 0, 6,
                                 5);
    const auto token = reconstruction_.expected_initial_projection_token();
    mark_projection_rows(execution, reconstruction_projected_,
                         reconstruction_projected_stamps_,
                         reconstruction_.current_coefficient_stamps(), 0, 7,
                         5, generation_, token);
    reconstruction_.accept_initial_projection(
        execution, reconstruction_projected_,
        reconstruction_projected_stamps_, generation_, token);
  }

  void advance_one_level(const execution_space& execution,
                         const std::uint64_t generation) {
    if (!initialized_ || generation != generation_ || current_level() >= 4 ||
        reconstruction_.current_level() != current_level()) {
      throw std::logic_error(
          "Route-B angular coordinator level is unavailable");
    }
    const std::size_t level = current_level();
    const std::size_t next_active = 4 - level;
    Kokkos::deep_copy(execution, primary_snapshot_,
                      primary_.psi_coefficients());
    Kokkos::deep_copy(execution, primary_snapshot_stamps_,
                      primary_.psi_coefficient_stamps());
    Kokkos::deep_copy(execution, primary_laplacian_, Complex{});
    Kokkos::deep_copy(execution, primary_laplacian_stamps_, std::uint64_t{0});
    for (std::size_t order = 0; order < next_active; ++order) {
      first_angular_.laplacian(execution, primary_snapshot_, order,
                               primary_laplacian_, order);
    }
    mark_rank4_rows(execution, primary_laplacian_,
                    primary_laplacian_stamps_, primary_snapshot_stamps_,
                    primary_snapshot_stamps_, 0, next_active, generation_,
                    generation_, generation_, false);
    primary_.advance(execution, primary_laplacian_,
                     primary_laplacian_stamps_, generation_,
                     ReductionEvolution::FreeDamped, 0.0);
    project_primary_current(execution);

    const auto token1 = reconstruction_.expected_pass_token();
    Kokkos::deep_copy(execution, reconstruction_angular_, Complex{});
    Kokkos::deep_copy(execution, reconstruction_angular_stamps_,
                      std::uint64_t{0});
    for (std::size_t order = 0; order < next_active; ++order) {
      first_angular_.pure_raise(execution, primary_snapshot_, order,
                                reconstruction_pure_, order);
    }
    apply_ghp_jets(execution, primary_snapshot_, primary_.psi_coefficients(),
                   reconstruction_pure_, reconstruction_angular_, -2, -2,
                   false, next_active);
    mark_rank4_rows(execution, reconstruction_angular_,
                    reconstruction_angular_stamps_, primary_snapshot_stamps_,
                    primary_.psi_coefficient_stamps(), 0, next_active,
                    generation_, generation_, token1, true);
    reconstruction_.pass1(
        execution, primary_snapshot_, primary_snapshot_stamps_,
        reconstruction_angular_, reconstruction_angular_stamps_, generation_,
        token1, ReductionEvolution::FreeDamped, 0.0);
    project_reconstruction_range(execution,
                                 reconstruction_.next_coefficients(), 0, 1,
                                 next_active);
    mark_projection_rows(execution, reconstruction_projected_,
                         reconstruction_projected_stamps_,
                         reconstruction_.next_coefficient_stamps(), 0, 2,
                         next_active, token1, token1);
    reconstruction_.accept_pass1_projection(
        execution, reconstruction_projected_,
        reconstruction_projected_stamps_, generation_, token1);

    const auto token2 = reconstruction_.expected_pass_token();
    Kokkos::deep_copy(execution, reconstruction_angular_, Complex{});
    Kokkos::deep_copy(execution, reconstruction_angular_stamps_,
                      std::uint64_t{0});
    const auto current_coefficients = reconstruction_.current_coefficients();
    const auto next_coefficients = reconstruction_.next_coefficients();
    for (std::size_t order = 0; order < next_active; ++order) {
      const auto current = Kokkos::subview(
          current_coefficients, Kokkos::ALL, Kokkos::ALL,
          order, Kokkos::ALL, Kokkos::ALL);
      g_angular_.pure_raise(execution, current, 0, reconstruction_pure_,
                            order);
    }
    const auto current_g = Kokkos::subview(
        current_coefficients, Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL);
    const auto next_g = Kokkos::subview(
        next_coefficients, Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL);
    apply_ghp_jets(execution, current_g, next_g, reconstruction_pure_,
                   reconstruction_angular_, -1, -1, false, next_active);
    const auto current_g_stamps = Kokkos::subview(
        reconstruction_.current_coefficient_stamps(), Kokkos::ALL, 0,
        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    const auto next_g_stamps = Kokkos::subview(
        reconstruction_.next_coefficient_stamps(), Kokkos::ALL, 0,
        Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    mark_rank4_rows(execution, reconstruction_angular_,
                    reconstruction_angular_stamps_, current_g_stamps,
                    next_g_stamps, 0, next_active, generation_, token1,
                    token2, true);
    reconstruction_.pass2(execution, reconstruction_angular_,
                          reconstruction_angular_stamps_, generation_, token2,
                          ReductionEvolution::FreeDamped, 0.0);
    project_reconstruction_range(execution,
                                 reconstruction_.next_coefficients(), 2, 5,
                                 next_active);
    mark_projection_rows(execution, reconstruction_projected_,
                         reconstruction_projected_stamps_,
                         reconstruction_.next_coefficient_stamps(), 2, 4,
                         next_active, token2, token2);
    reconstruction_.accept_pass2_projection(
        execution, reconstruction_projected_,
        reconstruction_projected_stamps_, generation_, token2);

    const auto token3 = reconstruction_.expected_pass_token();
    Kokkos::deep_copy(execution, reconstruction_pass3_, Complex{});
    Kokkos::deep_copy(execution, reconstruction_pass3_stamps_,
                      std::uint64_t{0});
    pack_sharp_pairs(execution, next_active);
    apply_reconstruction_pass3_angular(execution, next_active);
    mark_pass3_rows(execution, next_active, token1, token2, token3);
    reconstruction_.pass3(execution, reconstruction_pass3_,
                          reconstruction_pass3_stamps_, generation_, token3,
                          ReductionEvolution::FreeDamped, 0.0);
    project_reconstruction_range(execution,
                                 reconstruction_.current_coefficients(), 6, 6,
                                 next_active);
    mark_projection_rows(execution, reconstruction_projected_,
                         reconstruction_projected_stamps_,
                         reconstruction_.current_coefficient_stamps(), 6, 1,
                         next_active, generation_, token3);
    reconstruction_.accept_pass3_projection(
        execution, reconstruction_projected_,
        reconstruction_projected_stamps_, generation_, token3);
  }

  void advance_to_h4(const execution_space& execution,
                     const std::uint64_t generation) {
    while (current_level() < 4) advance_one_level(execution, generation);
  }

 private:
  [[nodiscard]] static int checked_node_count(
      const ModeRegistry& registry, const UniformRadialGrid& grid,
      const int ell_max, const int node_count,
      const TeukolskyParameters& parameters) {
    if (!registry.is_closed_under_sharp() || registry.size() == 0 ||
        grid.size() < routeb_fornberg_window || ell_max < 2 ||
        node_count <= 0 || node_count < ell_max + 1 ||
        parameters.spin_weight != -2 || parameters.azimuthal_mode != 0 ||
        !(parameters.mass > 0.0) || !std::isfinite(parameters.mass) ||
        !std::isfinite(parameters.spin) ||
        std::abs(parameters.spin) > parameters.mass ||
        !(parameters.compactification_length > 0.0) ||
        !std::isfinite(parameters.compactification_length) ||
        !std::isfinite(parameters.reduction_damping) ||
        parameters.reduction_damping < 0.0) {
      throw std::invalid_argument(
          "Route-B angular coordinator configuration invalid");
    }
    for (const int mode : registry.modes()) {
      if (std::abs(mode) > ell_max) {
        throw std::invalid_argument(
            "Route-B angular coordinator mode is outside its band");
      }
    }
    return node_count;
  }

  template <class PrimaryInput, class PrimaryStampView,
            class ReconstructionInput, class ReconstructionStampView>
  void prevalidate_initial(
      const execution_space& execution, const PrimaryInput& primary_input,
      const PrimaryStampView& primary_stamps,
      const ReconstructionInput& reconstruction_input,
      const ReconstructionStampView& reconstruction_stamps,
      const std::uint64_t generation) {
    static_assert(PrimaryInput::rank == 4 && PrimaryStampView::rank == 3 &&
                  ReconstructionInput::rank == 4 &&
                  ReconstructionStampView::rank == 3);
    static_assert(
        std::is_same_v<typename PrimaryInput::non_const_value_type, Complex> &&
        std::is_same_v<typename ReconstructionInput::non_const_value_type,
                       Complex> &&
        std::is_same_v<typename PrimaryStampView::non_const_value_type,
                       std::uint64_t> &&
        std::is_same_v<
            typename ReconstructionStampView::non_const_value_type,
            std::uint64_t>);
    static_assert(Kokkos::SpaceAccessibility<
                      execution_space,
                      typename PrimaryInput::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename PrimaryStampView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ReconstructionInput::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<
                      execution_space,
                      typename ReconstructionStampView::memory_space>::accessible);
    const auto valid_input = [&](const auto& input, const std::size_t fields) {
      return input.data() != nullptr && input.extent(0) == registry_.size() &&
             input.extent(1) == fields && input.extent(2) == grid_.size() &&
             input.extent(3) == static_cast<std::size_t>(node_count_) &&
             routeb_fornberg_detail::has_separated_strides<4>(input);
    };
    const auto valid_stamps = [&](const auto& stamps) {
      return stamps.data() != nullptr &&
             stamps.extent(0) == registry_.size() &&
             stamps.extent(1) == grid_.size() &&
             stamps.extent(2) == static_cast<std::size_t>(node_count_) &&
             routeb_fornberg_detail::has_separated_strides<3>(stamps);
    };
    const bool external_overlap =
        routeb_fornberg_detail::allocations_overlap(primary_input,
                                                     primary_stamps) ||
        routeb_fornberg_detail::allocations_overlap(primary_input,
                                                     reconstruction_input) ||
        routeb_fornberg_detail::allocations_overlap(primary_input,
                                                     reconstruction_stamps) ||
        routeb_fornberg_detail::allocations_overlap(primary_stamps,
                                                     reconstruction_input) ||
        routeb_fornberg_detail::allocations_overlap(primary_stamps,
                                                     reconstruction_stamps) ||
        routeb_fornberg_detail::allocations_overlap(reconstruction_input,
                                                     reconstruction_stamps);
    if (!valid_input(primary_input, 3) || !valid_stamps(primary_stamps) ||
        !valid_input(reconstruction_input, 7) ||
        !valid_stamps(reconstruction_stamps) ||
        !RouteBTeukolskyPrimaryJetTower<execution_space>::
            generation_supported(generation) ||
        !RouteBReconstructionJetTower<execution_space>::generation_supported(
            generation) ||
        external_overlap || overlaps_owned(primary_input) ||
        overlaps_owned(primary_stamps) ||
        overlaps_owned(reconstruction_input) ||
        overlaps_owned(reconstruction_stamps)) {
      throw std::invalid_argument(
          "Route-B angular coordinator initial views invalid");
    }
    (void)execution;
    (void)generation;
  }

  template <class View>
  [[nodiscard]] bool overlaps_owned(const View& view) const {
    return routeb_fornberg_detail::allocations_overlap(view,
                                                        primary_.values()) ||
           routeb_fornberg_detail::allocations_overlap(view,
                                                        primary_.stamps()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_.current_coefficients()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_.current_coefficient_stamps()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_.values()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_.stamps()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_.current_coefficients()) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_.current_coefficient_stamps()) ||
           routeb_fornberg_detail::allocations_overlap(view, modes_) ||
           routeb_fornberg_detail::allocations_overlap(view, sharp_) ||
           routeb_fornberg_detail::allocations_overlap(view, theta_) ||
           routeb_fornberg_detail::allocations_overlap(view, radius_) ||
           routeb_fornberg_detail::allocations_overlap(view, sin_theta_) ||
           routeb_fornberg_detail::allocations_overlap(view, cos_theta_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_snapshot_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_snapshot_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_laplacian_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_laplacian_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_projected_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, primary_projected_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_angular_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_pure_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_angular_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_pass3_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_pass3_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_projected_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, reconstruction_projected_stamps_) ||
           routeb_fornberg_detail::allocations_overlap(view, sharp_pairs_) ||
           routeb_fornberg_detail::allocations_overlap(
               view, sharp_pair_stamps_);
  }

  void project_primary_current(const execution_space& execution) {
    const std::size_t active = 5 - primary_.current_level();
    Kokkos::deep_copy(execution, primary_projected_, Complex{});
    Kokkos::deep_copy(execution, primary_projected_stamps_, std::uint64_t{0});
    for (std::size_t order = 0; order < active; ++order) {
      const auto current = Kokkos::subview(
          primary_.current_coefficients(), Kokkos::ALL, Kokkos::ALL, order,
          Kokkos::ALL, Kokkos::ALL);
      const auto projected = Kokkos::subview(
          primary_projected_, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
          Kokkos::ALL);
      for (std::size_t field = 0; field < 3; ++field) {
        first_angular_.project(execution, current, field, projected, field);
      }
    }
    const auto token = primary_.expected_projection_token();
    mark_projection_rows(execution, primary_projected_,
                         primary_projected_stamps_,
                         primary_.current_coefficient_stamps(), 0, 3, active,
                         generation_, token);
    primary_.accept_projected_current(execution, primary_projected_,
                                      primary_projected_stamps_, generation_,
                                      token);
  }

  template <class FieldView, class DtView, class PureView, class OutputView>
  void apply_ghp_jets(const execution_space& execution,
                      const FieldView& field, const DtView& dt,
                      const PureView& pure, const OutputView& output,
                      const int spin, const int boost, const bool prime,
                      const std::size_t active) {
    const std::size_t total = registry_.size() * grid_.size() *
                              static_cast<std::size_t>(node_count_);
    Kokkos::parallel_for(
        prime ? "routeb_apply_ethprime_radial_jets"
              : "routeb_apply_eth_radial_jets",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_angular_jet_detail::ApplyGhpJetFunctor{
            field.data(), dt.data(), pure.data(), output.data(),
            radius_.data(),
            {field.stride(0), field.stride(1), field.stride(2),
             field.stride(3)},
            {dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3)},
            {pure.stride(0), pure.stride(1), pure.stride(2), pure.stride(3)},
            {output.stride(0), output.stride(1), output.stride(2),
             output.stride(3)},
            grid_.size(), static_cast<std::size_t>(node_count_), active, spin,
            boost, parameters_.spin, parameters_.compactification_length,
            sin_theta_.data(), cos_theta_.data(), prime});
  }

  template <class OutputView, class OutputStampView, class SourceAView,
            class SourceBView>
  void mark_rank4_rows(const execution_space& execution,
                       const OutputView& output,
                       const OutputStampView& output_stamps,
                       const SourceAView& source_a,
                       const SourceBView& source_b,
                       const std::size_t first_order,
                       const std::size_t active_orders,
                       const std::uint64_t source_a_token,
                       const std::uint64_t source_b_token,
                       const std::uint64_t output_token,
                       const bool use_source_b) {
    const std::size_t total =
        registry_.size() * active_orders * grid_.size();
    Kokkos::parallel_for(
        "routeb_mark_rank4_angular_rows",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_angular_jet_detail::MarkRank4Rows{
            output.data(), output_stamps.data(), source_a.data(),
            source_b.data(),
            {output.stride(0), output.stride(1), output.stride(2),
             output.stride(3)},
            {output_stamps.stride(0), output_stamps.stride(1),
             output_stamps.stride(2), output_stamps.stride(3)},
            {source_a.stride(0), source_a.stride(1), source_a.stride(2),
             source_a.stride(3)},
            {source_b.stride(0), source_b.stride(1), source_b.stride(2),
             source_b.stride(3)},
            grid_.size(), static_cast<std::size_t>(node_count_), first_order,
            active_orders, source_a_token, source_b_token, output_token,
            use_source_b});
  }

  template <class OutputView, class OutputStampView, class SourceStampView>
  void mark_projection_rows(const execution_space& execution,
                            const OutputView& output,
                            const OutputStampView& output_stamps,
                            const SourceStampView& source_stamps,
                            const std::size_t first_field,
                            const std::size_t field_count,
                            const std::size_t active_orders,
                            const std::uint64_t source_token,
                            const std::uint64_t output_token) {
    const std::size_t total = registry_.size() * field_count * active_orders *
                              grid_.size();
    Kokkos::parallel_for(
        "routeb_mark_projected_rows",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        routeb_angular_jet_detail::MarkRank5Rows{
            output.data(), output_stamps.data(), source_stamps.data(),
            {output.stride(0), output.stride(1), output.stride(2),
             output.stride(3), output.stride(4)},
            {output_stamps.stride(0), output_stamps.stride(1),
             output_stamps.stride(2), output_stamps.stride(3),
             output_stamps.stride(4)},
            {source_stamps.stride(0), source_stamps.stride(1),
             source_stamps.stride(2), source_stamps.stride(3),
             source_stamps.stride(4)},
            grid_.size(), static_cast<std::size_t>(node_count_), first_field,
            field_count, active_orders, source_token, output_token});
  }

  void mark_pass3_rows(const execution_space& execution,
                       const std::size_t active, const std::uint64_t token1,
                       const std::uint64_t token2,
                       const std::uint64_t token3) {
    const auto current_stamps = reconstruction_.current_coefficient_stamps();
    const auto next_stamps = reconstruction_.next_coefficient_stamps();
    for (const auto descriptor :
         {std::array<std::size_t, 3>{0, 5, 5},
          std::array<std::size_t, 3>{1, 4, 4}}) {
      const auto output = Kokkos::subview(
          reconstruction_pass3_, Kokkos::ALL, descriptor[0], Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      const auto output_stamps = Kokkos::subview(
          reconstruction_pass3_stamps_, Kokkos::ALL, descriptor[0],
          Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      const auto source_a = Kokkos::subview(
          current_stamps, Kokkos::ALL, descriptor[1], Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      const auto source_b = Kokkos::subview(
          next_stamps, Kokkos::ALL, descriptor[2], Kokkos::ALL, Kokkos::ALL,
          Kokkos::ALL);
      mark_rank4_rows(execution, output, output_stamps, source_a, source_b, 0,
                      active, generation_, token2, token3, true);
    }
    for (std::size_t slot = 2; slot < 4; ++slot) {
      const std::size_t value_slot = slot == 2 ? 0 : 2;
      const std::size_t dt_slot = value_slot + 1;
      const auto output = Kokkos::subview(
          reconstruction_pass3_, Kokkos::ALL, slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      const auto output_stamps = Kokkos::subview(
          reconstruction_pass3_stamps_, Kokkos::ALL, slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      const auto source_a = Kokkos::subview(
          sharp_pair_stamps_, Kokkos::ALL, value_slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      const auto source_b = Kokkos::subview(
          sharp_pair_stamps_, Kokkos::ALL, dt_slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      mark_rank4_rows(execution, output, output_stamps, source_a, source_b, 0,
                      active, generation_, token2, token3, true);
    }
    (void)token1;
  }

  void apply_reconstruction_pass3_angular(const execution_space& execution,
                                          const std::size_t active) {
    const auto current = reconstruction_.current_coefficients();
    const auto next = reconstruction_.next_coefficients();
    for (std::size_t order = 0; order < active; ++order) {
      const auto current_order = Kokkos::subview(
          current, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
          Kokkos::ALL);
      const auto sharp_order = Kokkos::subview(
          sharp_pairs_, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
          Kokkos::ALL);
      c_angular_.pure_raise(execution, current_order, 5,
                            reconstruction_pure_, order);
    }
    auto apply = [&](const std::size_t field, const int spin, const int boost,
                     const std::size_t slot) {
      const auto field_view = Kokkos::subview(
          current, Kokkos::ALL, field, Kokkos::ALL, Kokkos::ALL,
          Kokkos::ALL);
      const auto dt_view = Kokkos::subview(
          next, Kokkos::ALL, field, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      const auto output = Kokkos::subview(
          reconstruction_pass3_, Kokkos::ALL, slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      apply_ghp_jets(execution, field_view, dt_view, reconstruction_pure_,
                     output, spin, boost, false, active);
    };
    apply(5, -1, 1, 0);
    for (std::size_t order = 0; order < active; ++order) {
      const auto current_order = Kokkos::subview(
          current, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
          Kokkos::ALL);
      pi_angular_.pure_raise(execution, current_order, 4,
                             reconstruction_pure_, order);
    }
    apply(4, -1, 0, 1);
    auto apply_sharp = [&](const std::size_t value_slot,
                           const std::size_t dt_slot, const int spin,
                           const int boost, const std::size_t output_slot,
                           auto& angular) {
      for (std::size_t order = 0; order < active; ++order) {
        const auto sharp_order = Kokkos::subview(
            sharp_pairs_, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
            Kokkos::ALL);
        angular.pure_lower(execution, sharp_order, value_slot,
                           reconstruction_pure_, order);
      }
      const auto value = Kokkos::subview(
          sharp_pairs_, Kokkos::ALL, value_slot, Kokkos::ALL, Kokkos::ALL,
          Kokkos::ALL);
      const auto dt = Kokkos::subview(
          sharp_pairs_, Kokkos::ALL, dt_slot, Kokkos::ALL, Kokkos::ALL,
          Kokkos::ALL);
      const auto output = Kokkos::subview(
          reconstruction_pass3_, Kokkos::ALL, output_slot, Kokkos::ALL,
          Kokkos::ALL, Kokkos::ALL);
      apply_ghp_jets(execution, value, dt, reconstruction_pure_, output, spin,
                     boost, true, active);
    };
    apply_sharp(0, 1, 2, 0, 2, b_sharp_angular_);
    apply_sharp(2, 3, 1, 1, 3, c_sharp_angular_);
  }

  void project_reconstruction_range(const execution_space& execution,
                                    const auto& coefficients,
                                    const std::size_t first_field,
                                    const std::size_t last_field,
                                    const std::size_t active) {
    Kokkos::deep_copy(execution, reconstruction_projected_, Complex{});
    Kokkos::deep_copy(execution, reconstruction_projected_stamps_,
                      std::uint64_t{0});
    for (std::size_t order = 0; order < active; ++order) {
      const auto current = Kokkos::subview(
          coefficients, Kokkos::ALL, Kokkos::ALL, order, Kokkos::ALL,
          Kokkos::ALL);
      const auto projected = Kokkos::subview(
          reconstruction_projected_, Kokkos::ALL, Kokkos::ALL, order,
          Kokkos::ALL, Kokkos::ALL);
      for (std::size_t field = first_field; field <= last_field; ++field) {
        project_reconstruction_field(execution, field, current, projected);
      }
    }
  }

  template <class InputView, class OutputView>
  void project_reconstruction_field(const execution_space& execution,
                                    const std::size_t field,
                                    const InputView& input,
                                    const OutputView& output) {
    if (field == 0) {
      g_angular_.project(execution, input, field, output, field);
    } else if (field == 1 || field == 3) {
      first_angular_.project(execution, input, field, output, field);
    } else if (field == 2 || field == 6) {
      scalar_angular_.project(execution, input, field, output, field);
    } else if (field == 4) {
      pi_angular_.project(execution, input, field, output, field);
    } else if (field == 5) {
      c_angular_.project(execution, input, field, output, field);
    }
  }

  void pack_sharp_pairs(const execution_space& execution,
                        const std::size_t active) {
    const auto current = reconstruction_.current_coefficients();
    const auto next = reconstruction_.next_coefficients();
    const auto sharp = sharp_;
    const auto output = sharp_pairs_;
    const auto output_stamps = sharp_pair_stamps_;
    const auto current_stamps = reconstruction_.current_coefficient_stamps();
    const auto next_stamps = reconstruction_.next_coefficient_stamps();
    const std::size_t radial_count = grid_.size();
    const std::size_t theta_count = static_cast<std::size_t>(node_count_);
    const std::size_t total = registry_.size() * 4 * radial_count * theta_count;
    Kokkos::parallel_for(
        "routeb_pack_sharp_coefficient_pairs",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = 4 * radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within = flat - mode * plane;
          const std::size_t order = within / (radial_count * theta_count);
          const std::size_t point =
              within - order * radial_count * theta_count;
          const std::size_t radial = point / theta_count;
          const std::size_t theta = point - radial * theta_count;
          if (order < active) {
            const std::size_t partner = sharp(mode);
            output(mode, 0, order, radial, theta) =
                Kokkos::conj(current(partner, 3, order, radial, theta));
            output(mode, 1, order, radial, theta) =
                Kokkos::conj(next(partner, 3, order, radial, theta));
            output(mode, 2, order, radial, theta) =
                Kokkos::conj(current(partner, 5, order, radial, theta));
            output(mode, 3, order, radial, theta) =
                Kokkos::conj(next(partner, 5, order, radial, theta));
            output_stamps(mode, 0, order, radial, theta) =
                current_stamps(partner, 3, order, radial, theta);
            output_stamps(mode, 1, order, radial, theta) =
                next_stamps(partner, 3, order, radial, theta);
            output_stamps(mode, 2, order, radial, theta) =
                current_stamps(partner, 5, order, radial, theta);
            output_stamps(mode, 3, order, radial, theta) =
                next_stamps(partner, 5, order, radial, theta);
          } else {
            for (std::size_t slot = 0; slot < 4; ++slot) {
              output(mode, slot, order, radial, theta) = {};
              output_stamps(mode, slot, order, radial, theta) = 0;
            }
          }
        });
  }

  ModeRegistry registry_;
  UniformRadialGrid grid_;
  int ell_max_;
  int node_count_;
  TeukolskyParameters parameters_;
  KerrParameters background_;
  Kokkos::View<int*, memory_space> modes_;
  Kokkos::View<std::size_t*, memory_space> sharp_;
  Kokkos::View<Real*, memory_space> theta_;
  Kokkos::View<Real*, memory_space> radius_;
  Kokkos::View<Real*, memory_space> sin_theta_;
  Kokkos::View<Real*, memory_space> cos_theta_;
  RouteBTeukolskyPrimaryJetTower<execution_space> primary_;
  RouteBReconstructionJetTower<execution_space> reconstruction_;
  SignedModeAngularCoordinator<execution_space> first_angular_;
  SignedModeAngularCoordinator<execution_space> g_angular_;
  SignedModeAngularCoordinator<execution_space> pi_angular_;
  SignedModeAngularCoordinator<execution_space> c_angular_;
  SignedModeAngularCoordinator<execution_space> scalar_angular_;
  SignedModeAngularCoordinator<execution_space> b_sharp_angular_;
  SignedModeAngularCoordinator<execution_space> c_sharp_angular_;
  complex4 primary_snapshot_;
  stamp4 primary_snapshot_stamps_;
  complex4 primary_laplacian_;
  stamp4 primary_laplacian_stamps_;
  complex5 primary_projected_;
  stamp5 primary_projected_stamps_;
  complex4 reconstruction_angular_;
  complex4 reconstruction_pure_;
  stamp4 reconstruction_angular_stamps_;
  complex5 reconstruction_pass3_;
  stamp5 reconstruction_pass3_stamps_;
  complex5 reconstruction_projected_;
  stamp5 reconstruction_projected_stamps_;
  complex5 sharp_pairs_;
  stamp5 sharp_pair_stamps_;
  std::uint64_t generation_ = 0;
  bool initialized_ = false;
};

}  // namespace teuk
