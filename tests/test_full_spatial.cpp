#include "test_harness.hpp"

#include <cstddef>

#include "teuk/full_spatial.hpp"
#include "teuk/linear_spatial.hpp"
#include "teuk/modes.hpp"
#include "teuk/reconstruction_spatial.hpp"

namespace {

constexpr std::size_t teuk_fields =
    static_cast<std::size_t>(teuk::TeukolskyField::Count);
constexpr std::size_t teuk_scratch =
    static_cast<std::size_t>(teuk::TeukolskyRadialScratch::Count);
constexpr std::size_t reconstruction_fields =
    static_cast<std::size_t>(teuk::ReconstructionField::Count);
constexpr std::size_t reconstruction_angular =
    static_cast<std::size_t>(teuk::ReconstructionAngularInput::Count);
constexpr std::size_t pipeline_fields = 13;

}  // namespace

TEST_CASE("full Teukolsky first and second stages equal every fixed-theta slice") {
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr std::size_t theta_count = 3;
  teuk::FullSpatialThetaView theta("full_teuk_theta", theta_count);
  teuk::SignedModeView modes("full_teuk_modes", registry.size());
  const std::size_t full_size =
      registry.size() * pipeline_fields * grid.size() * theta_count;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> state_flat(
      "full_teuk_state_flat", full_size);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> first_rhs_flat(
      "full_teuk_first_rhs_flat", full_size);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> second_rhs_flat(
      "full_teuk_second_rhs_flat", full_size);
  using UnmanagedFullView =
      Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  UnmanagedFullView state(state_flat.data(), registry.size(), pipeline_fields,
                          grid.size(), theta_count);
  teuk::FullSpatialValueView angular("full_teuk_angular", registry.size(),
                                      grid.size(), theta_count);
  teuk::FullSpatialValueView zero_forcing(
      "full_teuk_zero_forcing", registry.size(), grid.size(), theta_count);
  teuk::FullSpatialValueView driven_forcing(
      "full_teuk_driven_forcing", registry.size(), grid.size(), theta_count);
  UnmanagedFullView first_rhs(first_rhs_flat.data(), registry.size(),
                              pipeline_fields, grid.size(), theta_count);
  UnmanagedFullView second_rhs(second_rhs_flat.data(), registry.size(),
                               pipeline_fields, grid.size(), theta_count);
  teuk::FullSpatialStateView first_scratch(
      "full_teuk_first_scratch", registry.size(), teuk_scratch, grid.size(),
      theta_count);
  teuk::FullSpatialStateView second_scratch(
      "full_teuk_second_scratch", registry.size(), teuk_scratch, grid.size(),
      theta_count);

  auto host_theta = Kokkos::create_mirror_view(theta);
  auto host_modes = Kokkos::create_mirror_view(modes);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_state("full_teuk_host_state", registry.size(), pipeline_fields,
                 grid.size(), theta_count);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_angular("full_teuk_host_angular", registry.size(), grid.size(),
                   theta_count);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_forcing("full_teuk_host_forcing", registry.size(), grid.size(),
                   theta_count);
  for (std::size_t t = 0; t < theta_count; ++t) host_theta(t) = 0.43 + 0.31 * t;
  const teuk::TeukolskyFullFieldOffsets first_fields{0, 1, 2};
  const teuk::TeukolskyFullFieldOffsets second_fields{10, 11, 12};
  for (std::size_t m = 0; m < registry.size(); ++m) {
    host_modes(m) = registry.modes()[m];
    for (std::size_t r = 0; r < grid.size(); ++r) {
      const double radius = grid.coordinate(r);
      for (std::size_t t = 0; t < theta_count; ++t) {
        for (std::size_t f = 0; f < teuk_fields; ++f) {
          const teuk::Complex value = teuk::Complex(
              0.1 * (m + 1) + 0.04 * (f + 1) * radius * radius + 0.03 * t,
              -0.06 * (f + 1) + 0.02 * (m + 1) * radius + 0.01 * t);
          host_state(m, f, r, t) = value;
          host_state(m, 10 + f, r, t) = 1.3 * value;
        }
        host_angular(m, r, t) = teuk::Complex(
            -0.2 * (m + 1) * radius + 0.05 * t,
            0.03 * (m + t + 1) - 0.04 * radius * radius);
        host_forcing(m, r, t) = teuk::Complex(
            0.07 * (m + 1) - 0.02 * radius + 0.01 * t,
            -0.05 * (t + 1) + 0.03 * radius);
      }
    }
  }
  Kokkos::deep_copy(theta, host_theta);
  Kokkos::deep_copy(modes, host_modes);
  Kokkos::deep_copy(state, host_state);
  Kokkos::deep_copy(angular, host_angular);
  Kokkos::deep_copy(zero_forcing, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(driven_forcing, host_forcing);

  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.42;
  parameters.compactification_length = 1.8;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.37;
  teuk::ExecutionSpace execution;
  teuk::evaluate_sbp_teukolsky_full_stage_rhs(
      execution, grid, parameters, theta, modes, state, angular, zero_forcing,
      teuk::ReductionEvolution::FreeDamped, first_scratch, first_rhs, 0.025,
      first_fields, first_fields);
  teuk::evaluate_sbp_teukolsky_full_stage_rhs(
      execution, grid, parameters, theta, modes, state, angular, driven_forcing,
      teuk::ReductionEvolution::FreeDamped, second_scratch, second_rhs, 0.025,
      second_fields, second_fields);
  execution.fence();
  const auto host_first = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), first_rhs);
  const auto host_second = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), second_rhs);

  for (std::size_t t = 0; t < theta_count; ++t) {
    teuk::TeukolskyRadialLines fixed(registry, grid,
                                      "fixed_teuk_theta_slice");
    auto fixed_state = Kokkos::create_mirror_view(fixed.state());
    auto fixed_angular = Kokkos::create_mirror_view(fixed.angular_laplacian());
    auto fixed_forcing = Kokkos::create_mirror_view(fixed.forcing());
    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t r = 0; r < grid.size(); ++r) {
        for (std::size_t f = 0; f < teuk_fields; ++f) {
          fixed_state(m, f, r) = host_state(m, f, r, t);
        }
        fixed_angular(m, r) = host_angular(m, r, t);
        fixed_forcing(m, r) = 0.0;
      }
    }
    Kokkos::deep_copy(fixed.state(), fixed_state);
    Kokkos::deep_copy(fixed.angular_laplacian(), fixed_angular);
    Kokkos::deep_copy(fixed.forcing(), fixed_forcing);
    teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
        fixed, parameters, host_theta(t), teuk::ReductionEvolution::FreeDamped,
        0.025);
    auto fixed_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         fixed.rhs());
    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t f = 0; f < teuk_fields; ++f) {
        for (std::size_t r = 0; r < grid.size(); ++r) {
          CHECK_COMPLEX_NEAR(host_first(m, f, r, t), fixed_rhs(m, f, r),
                             8.0e-13);
        }
      }
    }

    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t r = 0; r < grid.size(); ++r) {
        for (std::size_t f = 0; f < teuk_fields; ++f) {
          fixed_state(m, f, r) = host_state(m, 10 + f, r, t);
        }
        fixed_forcing(m, r) = host_forcing(m, r, t);
      }
    }
    Kokkos::deep_copy(fixed.state(), fixed_state);
    Kokkos::deep_copy(fixed.forcing(), fixed_forcing);
    teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
        fixed, parameters, host_theta(t), teuk::ReductionEvolution::FreeDamped,
        0.025);
    fixed_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    fixed.rhs());
    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t f = 0; f < teuk_fields; ++f) {
        for (std::size_t r = 0; r < grid.size(); ++r) {
          CHECK_COMPLEX_NEAR(host_second(m, 10 + f, r, t), fixed_rhs(m, f, r),
                             8.0e-13);
        }
      }
    }
  }

}

TEST_CASE("full reconstruction dependency passes equal every fixed-theta slice") {
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr std::size_t theta_count = 3;
  teuk::FullSpatialThetaView theta("full_reconstruction_theta", theta_count);
  teuk::FullSpatialStateView state(
      "full_reconstruction_state", registry.size(), reconstruction_fields,
      grid.size(), theta_count);
  teuk::FullSpatialValueView psi4("full_reconstruction_psi4", registry.size(),
                                   grid.size(), theta_count);
  teuk::FullSpatialStateView angular(
      "full_reconstruction_angular", registry.size(), reconstruction_angular,
      grid.size(), theta_count);
  teuk::FullSpatialValueView eth1_f("full_reconstruction_eth1f",
                                     registry.size(), grid.size(), theta_count);
  teuk::FullSpatialValueView eth2_g("full_reconstruction_eth2g",
                                     registry.size(), grid.size(), theta_count);
  teuk::FullSpatialStateView rhs("full_reconstruction_rhs", registry.size(),
                                  reconstruction_fields, grid.size(),
                                  theta_count);
  teuk::FullSpatialStateView dr("full_reconstruction_dr", registry.size(),
                                 reconstruction_fields, grid.size(),
                                 theta_count);
  teuk::ReconstructionRadialLines lookup(registry, grid,
                                          "full_reconstruction_lookup");

  auto host_theta = Kokkos::create_mirror_view(theta);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_state("full_reconstruction_host_state", registry.size(),
                 reconstruction_fields, grid.size(), theta_count);
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_psi4("full_reconstruction_host_psi4", registry.size(), grid.size(),
                theta_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_angular("full_reconstruction_host_angular", registry.size(),
                   reconstruction_angular, grid.size(), theta_count);
  for (std::size_t t = 0; t < theta_count; ++t) host_theta(t) = 0.38 + 0.34 * t;
  for (std::size_t m = 0; m < registry.size(); ++m) {
    for (std::size_t r = 0; r < grid.size(); ++r) {
      const double radius = grid.coordinate(r);
      for (std::size_t t = 0; t < theta_count; ++t) {
        host_psi4(m, r, t) = teuk::Complex(
            0.15 * (m + 1) + 0.04 * radius + 0.02 * t,
            -0.03 * (t + 1) + 0.01 * radius * radius);
        for (std::size_t f = 0; f < reconstruction_fields; ++f) {
          host_state(m, f, r, t) = teuk::Complex(
              0.03 * (m + 1) * (f + 1) + 0.02 * radius * radius + 0.01 * t,
              -0.02 * (f + 1) + 0.015 * (m + 1) * radius + 0.005 * t);
        }
        for (std::size_t a = 0; a < reconstruction_angular; ++a) {
          host_angular(m, a, r, t) = teuk::Complex(
              -0.04 * (a + 1) + 0.01 * (m + 1) * radius + 0.006 * t,
              0.02 * (m + a + 2) - 0.008 * radius * radius);
        }
      }
    }
  }
  Kokkos::deep_copy(theta, host_theta);
  Kokkos::deep_copy(state, host_state);
  Kokkos::deep_copy(psi4, host_psi4);
  Kokkos::deep_copy(angular, host_angular);
  Kokkos::parallel_for(
      "copy_full_reconstruction_eth1f",
      Kokkos::MDRangePolicy<teuk::ExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {registry.size(), grid.size(), theta_count}),
      KOKKOS_LAMBDA(const std::size_t m, const std::size_t r,
                    const std::size_t t) { eth1_f(m, r, t) = angular(m, 0, r, t); });
  Kokkos::deep_copy(rhs, teuk::Complex(123.0, -45.0));

  const teuk::KerrParameters parameters{1.0, 0.57, 1.7};
  teuk::ExecutionSpace execution;
  teuk::evaluate_sbp_reconstruction_full_radial_derivatives(execution, grid,
                                                             state, dr);
  teuk::evaluate_sbp_reconstruction_full_pass1(
      execution, grid, parameters, theta, lookup.sharp_indices(), state, psi4,
      eth1_f, dr, rhs);
  // The angular layer consumes the new dtG before pass 2.
  Kokkos::parallel_for(
      "form_full_reconstruction_eth2g",
      Kokkos::MDRangePolicy<teuk::ExecutionSpace, Kokkos::Rank<3>>(
          execution, {0, 0, 0},
          {registry.size(), grid.size(), theta_count}),
      KOKKOS_LAMBDA(const std::size_t m, const std::size_t r,
                    const std::size_t t) {
        eth2_g(m, r, t) = angular(m, 1, r, t) + 0.17 * rhs(m, 0, r, t);
        angular(m, 1, r, t) = eth2_g(m, r, t);
      });
  teuk::evaluate_sbp_reconstruction_full_pass2(
      execution, grid, parameters, theta, lookup.sharp_indices(), state, psi4,
      eth2_g, dr, rhs);
  // The angular layer now consumes new dtB/dtPi/dtC, including sharp modes.
  const auto sharp = lookup.sharp_indices();
  Kokkos::parallel_for(
      "form_full_reconstruction_pass3_angular",
      Kokkos::MDRangePolicy<teuk::ExecutionSpace, Kokkos::Rank<3>>(
          execution, {0, 0, 0},
          {registry.size(), grid.size(), theta_count}),
      KOKKOS_LAMBDA(const std::size_t m, const std::size_t r,
                    const std::size_t t) {
        constexpr std::size_t B =
            static_cast<std::size_t>(teuk::ReconstructionField::B);
        constexpr std::size_t Pi =
            static_cast<std::size_t>(teuk::ReconstructionField::Pi);
        constexpr std::size_t C =
            static_cast<std::size_t>(teuk::ReconstructionField::C);
        angular(m, 2, r, t) += 0.11 * rhs(m, C, r, t);
        angular(m, 3, r, t) += 0.13 * rhs(m, Pi, r, t);
        angular(m, 4, r, t) += 0.07 * Kokkos::conj(rhs(sharp(m), B, r, t));
        angular(m, 5, r, t) += 0.09 * Kokkos::conj(rhs(sharp(m), C, r, t));
      });
  teuk::evaluate_sbp_reconstruction_full_pass3(
      execution, grid, parameters, theta, lookup.sharp_indices(), state, psi4,
      angular, dr, rhs);
  execution.fence();

  const auto host_rhs =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs);
  const auto host_dr =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dr);
  const auto final_angular =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), angular);
  for (std::size_t t = 0; t < theta_count; ++t) {
    teuk::ReconstructionRadialLines fixed(registry, grid,
                                           "fixed_reconstruction_theta_slice");
    teuk::ReconstructionAngularInputView fixed_angular(
        "fixed_reconstruction_angular", registry.size(), reconstruction_angular,
        grid.size());
    auto fixed_state = Kokkos::create_mirror_view(fixed.state());
    auto fixed_psi4 = Kokkos::create_mirror_view(fixed.psi4());
    auto fixed_a = Kokkos::create_mirror_view(fixed_angular);
    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t r = 0; r < grid.size(); ++r) {
        fixed_psi4(m, r) = host_psi4(m, r, t);
        for (std::size_t f = 0; f < reconstruction_fields; ++f) {
          fixed_state(m, f, r) = host_state(m, f, r, t);
        }
        for (std::size_t a = 0; a < reconstruction_angular; ++a) {
          fixed_a(m, a, r) = final_angular(m, a, r, t);
        }
      }
    }
    Kokkos::deep_copy(fixed.state(), fixed_state);
    Kokkos::deep_copy(fixed.psi4(), fixed_psi4);
    Kokkos::deep_copy(fixed_angular, fixed_a);
    teuk::evaluate_sbp_reconstruction_radial_lines_rhs(
        fixed, parameters, host_theta(t), fixed_angular);
    const auto fixed_rhs = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fixed.rhs());
    const auto fixed_dr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fixed.radial_derivatives());
    for (std::size_t m = 0; m < registry.size(); ++m) {
      for (std::size_t f = 0; f < reconstruction_fields; ++f) {
        for (std::size_t r = 0; r < grid.size(); ++r) {
          CHECK_COMPLEX_NEAR(host_dr(m, f, r, t), fixed_dr(m, f, r),
                             7.0e-13);
          CHECK_COMPLEX_NEAR(host_rhs(m, f, r, t), fixed_rhs(m, f, r),
                             9.0e-13);
        }
      }
    }
  }

  const std::size_t full_size =
      registry.size() * pipeline_fields * grid.size() * theta_count;
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> stage13_flat(
      "reconstruction_stage13_flat", full_size);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> rhs13_flat(
      "reconstruction_rhs13_flat", full_size);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> dr13_flat(
      "reconstruction_dr13_flat", full_size);
  using UnmanagedFullView =
      Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  UnmanagedFullView stage13(stage13_flat.data(), registry.size(),
                            pipeline_fields, grid.size(), theta_count);
  UnmanagedFullView rhs13(rhs13_flat.data(), registry.size(), pipeline_fields,
                          grid.size(), theta_count);
  UnmanagedFullView dr13(dr13_flat.data(), registry.size(), pipeline_fields,
                         grid.size(), theta_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      host_stage13("reconstruction_host_stage13", registry.size(),
                   pipeline_fields, grid.size(), theta_count);
  Kokkos::deep_copy(host_stage13, teuk::Complex(0.0, 0.0));
  const teuk::ReconstructionFullFieldOffsets pipeline_reconstruction{
      3, 4, 5, 6, 7, 8, 9};
  for (std::size_t m = 0; m < registry.size(); ++m) {
    for (std::size_t r = 0; r < grid.size(); ++r) {
      for (std::size_t t = 0; t < theta_count; ++t) {
        host_stage13(m, 2, r, t) = host_psi4(m, r, t);
        for (std::size_t f = 0; f < reconstruction_fields; ++f) {
          host_stage13(m, 3 + f, r, t) = host_state(m, f, r, t);
        }
      }
    }
  }
  Kokkos::deep_copy(stage13, host_stage13);
  Kokkos::deep_copy(rhs13, teuk::Complex(0.0, 0.0));
  const auto psi4_subview =
      Kokkos::subview(stage13, Kokkos::ALL(), 2, Kokkos::ALL(), Kokkos::ALL());
  teuk::evaluate_sbp_reconstruction_full_radial_derivatives(
      execution, grid, stage13, dr13, pipeline_reconstruction,
      pipeline_reconstruction);
  teuk::evaluate_sbp_reconstruction_full_pass1(
      execution, grid, parameters, theta, lookup.sharp_indices(), stage13,
      psi4_subview, eth1_f, dr13, rhs13, pipeline_reconstruction,
      pipeline_reconstruction, pipeline_reconstruction);
  teuk::evaluate_sbp_reconstruction_full_pass2(
      execution, grid, parameters, theta, lookup.sharp_indices(), stage13,
      psi4_subview, eth2_g, dr13, rhs13, pipeline_reconstruction,
      pipeline_reconstruction, pipeline_reconstruction);
  teuk::evaluate_sbp_reconstruction_full_pass3(
      execution, grid, parameters, theta, lookup.sharp_indices(), stage13,
      psi4_subview, angular, dr13, rhs13, pipeline_reconstruction,
      pipeline_reconstruction, pipeline_reconstruction);
  execution.fence();
  const auto host_rhs13 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs13);
  const auto host_dr13 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dr13);
  for (std::size_t m = 0; m < registry.size(); ++m) {
    for (std::size_t f = 0; f < reconstruction_fields; ++f) {
      for (std::size_t r = 0; r < grid.size(); ++r) {
        for (std::size_t t = 0; t < theta_count; ++t) {
          CHECK_COMPLEX_NEAR(host_dr13(m, 3 + f, r, t), host_dr(m, f, r, t),
                             8.0e-13);
          CHECK_COMPLEX_NEAR(host_rhs13(m, 3 + f, r, t),
                             host_rhs(m, f, r, t), 1.0e-12);
        }
      }
    }
  }
}
