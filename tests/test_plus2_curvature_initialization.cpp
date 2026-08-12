#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "teuk/plus2_curvature_initialization.hpp"

namespace {

using C = teuk::Complex;

struct ManufacturedResult {
  teuk::Plus2PeelingResolutionSample sample;
  double maximum_error = 0.0;
};

ManufacturedResult manufactured(const std::size_t points,
                                const double kerr_spin,
                                const bool run_initializer,
                                const teuk::Plus2PeelingConvergenceCertificate*
                                    certificate = nullptr,
                                const bool change_nonleading_coefficient =
                                    false) {
  constexpr std::size_t modes = 2;
  constexpr std::size_t theta_count = 3;
  const std::vector<int> signed_modes{-2, 2};
  const std::vector<double> cos_theta_nodes{-0.6, 0.0, 0.6};
  const std::string profile_id =
      kerr_spin == 0.0 ? "analytic-numerator-Schwarzschild-v1"
                       : "analytic-numerator-rotating-Kerr-v1";
  const teuk::UniformRadialGrid grid(points, 0.0, 0.42);
  const double inverse_spacing = 1.0 / grid.spacing();
  std::vector<C> f0(modes * points * theta_count);
  std::vector<C> f1(f0.size());
  std::vector<C> c0(f0.size());
  std::vector<C> c1(f0.size());
  const auto flat = [&](const std::size_t mode, const std::size_t radial,
                        const std::size_t theta) {
    return (mode * points + radial) * theta_count + theta;
  };
  const auto amplitude = [](const std::size_t mode,
                            const std::size_t theta) {
    const C base(0.7 + 0.08 * static_cast<double>(theta),
                 -0.21 + 0.03 * static_cast<double>(theta));
    return mode == 0 ? base : Kokkos::conj(base);
  };
  const auto q0 = [&](const std::size_t mode, const std::size_t theta,
                      const double radius) {
    const double cos_theta = cos_theta_nodes[theta];
    const C denominator(1.7 * 1.7, -kerr_spin * radius * cos_theta);
    return amplitude(mode, theta) *
           Kokkos::exp(C(0.9 * radius, -0.23 * radius)) /
           (denominator * denominator * denominator * denominator);
  };
  const auto q1 = [&](const std::size_t mode, const std::size_t theta,
                      const double radius) {
    return (C(0.3, -0.17) + amplitude(mode, theta)) *
           Kokkos::exp(C(0.4 * radius, 0.31 * radius));
  };
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t radial = 0; radial < points; ++radial) {
      const double radius = grid.coordinate(radial);
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        const std::size_t i = flat(mode, radial, theta);
        const C coefficient_change =
            change_nonleading_coefficient && mode == 1 && theta == 2
                ? C(0.05, -0.02)
                : C{};
        f0[i] =
            radius * radius * (q0(mode, theta, radius) + coefficient_change);
        f1[i] = radius * q1(mode, theta, radius);
        // Smooth regular pieces exercise both fields and signed modes.
        c0[i] = amplitude(mode, theta) * C(0.04 + 0.02 * radius, 0.01);
        c1[i] = amplitude(mode, theta) * C(-0.03, 0.02 * radius);
      }
    }
  }

  double residual = 0.0;
  std::vector<C> z0_scri(modes * theta_count);
  std::vector<C> z1_scri(modes * theta_count);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      const std::size_t base = flat(mode, 0, theta);
      std::vector<C> df0(points);
      for (std::size_t radial = 0; radial < points; ++radial) {
        df0[radial] = teuk::radial_first_derivative_strided_at(
            teuk::RadialDiscretization::D105, f0.data() + base, points,
            radial, inverse_spacing, theta_count);
      }
      const C df0_scri = df0[0];
      residual = std::max(
          residual,
          std::max({Kokkos::abs(f0[base]), Kokkos::abs(df0_scri),
                    Kokkos::abs(f1[base])}));
      const std::size_t point = mode * theta_count + theta;
      z0_scri[point] = teuk::plus2_extract_q0_at_scri(
          f0.data() + base, points, inverse_spacing, theta_count);
      z1_scri[point] = teuk::plus2_extract_q1_at_scri(
          f1.data() + base, points, inverse_spacing, theta_count);
    }
  }

  ManufacturedResult result;
  result.sample = {points,
                   grid.lower_radius(),
                   grid.upper_radius(),
                   grid.spacing(),
                   residual,
                   signed_modes,
                   cos_theta_nodes,
                   profile_id,
                   z0_scri,
                   z1_scri};
  if (!run_initializer) return result;
  CHECK(certificate != nullptr);
  if (certificate == nullptr) return result;

  Kokkos::View<C***, Kokkos::LayoutRight> f0_view("f0", modes, points,
                                                  theta_count);
  Kokkos::View<C***, Kokkos::LayoutRight> f1_view("f1", modes, points,
                                                  theta_count);
  Kokkos::View<C***, Kokkos::LayoutRight> c0_view("c0", modes, points,
                                                  theta_count);
  Kokkos::View<C***, Kokkos::LayoutRight> c1_view("c1", modes, points,
                                                  theta_count);
  Kokkos::View<C****, Kokkos::LayoutRight> state("state", modes, 2, points,
                                                 theta_count);
  auto hf0 = Kokkos::create_mirror_view(f0_view);
  auto hf1 = Kokkos::create_mirror_view(f1_view);
  auto hc0 = Kokkos::create_mirror_view(c0_view);
  auto hc1 = Kokkos::create_mirror_view(c1_view);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t radial = 0; radial < points; ++radial) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        const auto i = flat(mode, radial, theta);
        hf0(mode, radial, theta) = f0[i];
        hf1(mode, radial, theta) = f1[i];
        hc0(mode, radial, theta) = c0[i];
        hc1(mode, radial, theta) = c1[i];
      }
    }
  }
  Kokkos::deep_copy(f0_view, hf0);
  Kokkos::deep_copy(f1_view, hf1);
  Kokkos::deep_copy(c0_view, hc0);
  Kokkos::deep_copy(c1_view, hc1);
  teuk::Plus2CurvatureInitializationWorkspace<> workspace(
      signed_modes, cos_theta_nodes, points);
  const teuk::Plus2CurvatureInitializationContract contract{
      teuk::Plus2CurvatureFormulaId::OrgRicciPeelingNumeratorsV1,
      teuk::Plus2PeelingEndpointOperatorId::ConstrainedPositiveNodesV2,
      teuk::Plus2RadialBoundaryPolicyId::
          ContinuumNoIncomingButWeaklyHyperbolicV1,
      profile_id, "manufactured-org-curvature-v1", *certificate,
      "manufactured-bianchi-residual-v1"};
  workspace.initialize(Kokkos::DefaultExecutionSpace{}, grid, f0_view,
                       f1_view, c0_view, c1_view, state, contract);
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        state);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t radial = 0; radial < points; ++radial) {
      const double radius = grid.coordinate(radial);
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        const auto i = flat(mode, radial, theta);
        result.maximum_error = std::max(
            result.maximum_error,
            Kokkos::abs(host(mode, 0, radial, theta) -
                        (q0(mode, theta, radius) + c0[i])));
        result.maximum_error = std::max(
            result.maximum_error,
            Kokkos::abs(host(mode, 1, radial, theta) -
                        (q1(mode, theta, radius) + c1[i])));
      }
    }
  }
  return result;
}

}  // namespace

TEST_CASE("plus2 Bianchi characteristics require no incoming radial data") {
  constexpr double mass = 1.0;
  constexpr double spin = 0.999;
  constexpr double length = 1.7;
  const teuk::KerrParameters parameters{mass, spin, length};
  const double r_plus = mass + std::sqrt(mass * mass - spin * spin);
  const double horizon = length * length / r_plus;
  const auto scri = teuk::plus2_bianchi_radial_characteristic(
      0.0, horizon, 0.8, 0.6, parameters);
  const auto interior = teuk::plus2_bianchi_radial_characteristic(
      0.4 * horizon, horizon, 0.8, 0.6, parameters);
  const auto at_horizon = teuk::plus2_bianchi_radial_characteristic(
      horizon, horizon, 0.8, 0.6, parameters);
  CHECK(scri.valid);
  CHECK(interior.valid);
  CHECK(at_horizon.valid);
  CHECK(scri.coordinate_speed == 0.0);
  CHECK(scri.kind ==
        teuk::Plus2BianchiRadialBoundaryKind::ScriCharacteristic);
  CHECK(scri.strongly_hyperbolic_radial_block);
  CHECK(interior.coordinate_speed > 0.0);
  CHECK(interior.kind ==
        teuk::Plus2BianchiRadialBoundaryKind::InteriorTowardHorizon);
  CHECK(!interior.strongly_hyperbolic_radial_block);
  CHECK(Kokkos::abs(interior.jordan_superdiagonal) > 0.0);
  const double radius = 0.4 * horizon;
  const double length2 = length * length;
  const double A = 2.0 + 4.0 * mass * radius / length2;
  const double BR = radius * radius / length2;
  const C c = -C(0.0, 1.0) * spin * 0.8 /
              (std::sqrt(2.0) * C(length2, -spin * radius * 0.6));
  CHECK_COMPLEX_NEAR(interior.jordan_superdiagonal, c * BR / (A * A),
                     2.0e-16);
  CHECK(at_horizon.coordinate_speed > 0.0);
  CHECK(at_horizon.kind ==
        teuk::Plus2BianchiRadialBoundaryKind::HorizonOutflow);
  CHECK(!at_horizon.strongly_hyperbolic_radial_block);
  const teuk::KerrParameters schwarzschild{mass, 0.0, length};
  CHECK(teuk::plus2_bianchi_radial_characteristic(
            0.4 * horizon, horizon, 0.8, 0.6, schwarzschild)
            .strongly_hyperbolic_radial_block);
  CHECK(!teuk::plus2_bianchi_radial_characteristic(
             -0.1, horizon, 0.8, 0.6, parameters)
             .valid);
}

TEST_CASE("plus2 peeling certificate rejects nonconvergent assertions") {
  const auto sample = [](const std::size_t points, const double residual,
                         const double z0) {
    const double spacing = 0.96 / static_cast<double>(points - 1);
    return teuk::Plus2PeelingResolutionSample{
        points,
        0.0,
        0.96,
        spacing,
        residual,
        {2},
        {0.0},
        "nonconvergent-profile-v1",
        {C(z0, 0.0)},
        {C(2.0, 0.0)}};
  };
  const auto coarse = sample(25, 1.0e-3, 1.01);
  const auto medium = sample(49, 7.0e-4, 1.02);
  const auto fine = sample(97, 6.0e-4, 1.03);
  bool rejected = false;
  try {
    static_cast<void>(teuk::qualify_plus2_peeling_convergence(
        coarse, medium, fine, "nonconvergent"));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("plus2 peeling certificate is bound before state mutation") {
  const auto coarse = manufactured(25, 0.73, false);
  const auto medium = manufactured(49, 0.73, false);
  const auto fine = manufactured(97, 0.73, false);
  const auto certificate = teuk::qualify_plus2_peeling_convergence(
      coarse.sample, medium.sample, fine.sample, "bound-peeling-v1", 4.0);
  constexpr std::size_t modes = 2;
  constexpr std::size_t theta = 3;
  constexpr std::size_t points = 97;
  const teuk::UniformRadialGrid grid(points, 0.0, 0.42);
  Kokkos::View<C***, Kokkos::LayoutRight> f0("bad_f0", modes, points,
                                             theta);
  Kokkos::View<C***, Kokkos::LayoutRight> zero("zero_regular", modes,
                                               points, theta);
  Kokkos::View<C****, Kokkos::LayoutRight> state("sentinel_state", modes, 2,
                                                 points, theta);
  Kokkos::deep_copy(f0, C(0.1, -0.2));
  Kokkos::deep_copy(zero, C{});
  Kokkos::deep_copy(state, C(9.0, -4.0));
  teuk::Plus2CurvatureInitializationWorkspace<> workspace(
      {-2, 2}, {-0.6, 0.0, 0.6}, points);
  const teuk::Plus2CurvatureInitializationContract contract{
      teuk::Plus2CurvatureFormulaId::OrgRicciPeelingNumeratorsV1,
      teuk::Plus2PeelingEndpointOperatorId::ConstrainedPositiveNodesV2,
      teuk::Plus2RadialBoundaryPolicyId::
          ContinuumNoIncomingButWeaklyHyperbolicV1,
      fine.sample.profile_id, "bad-data-must-not-mutate", certificate,
      "bad-data-must-not-mutate-bianchi"};
  bool rejected = false;
  try {
    workspace.initialize(Kokkos::DefaultExecutionSpace{}, grid, f0, zero,
                         zero, zero, state, contract);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        state);
  CHECK_COMPLEX_NEAR(host(0, 0, 0, 0), C(9.0, -4.0), 0.0);
}

TEST_CASE("plus2 peeling certificate binds every mode and angular node") {
  const auto coarse = manufactured(25, 0.73, false);
  const auto medium = manufactured(49, 0.73, false);
  const auto fine = manufactured(97, 0.73, false);
  const auto certificate = teuk::qualify_plus2_peeling_convergence(
      coarse.sample, medium.sample, fine.sample,
      "complete-endpoint-plane-v1", 4.0);
  const auto changed = manufactured(97, 0.73, false, nullptr, true);
  CHECK_NEAR(changed.sample.peeling_residual_norm,
             fine.sample.peeling_residual_norm, 0.0);
  CHECK_COMPLEX_NEAR(changed.sample.z0_scri[0], fine.sample.z0_scri[0], 0.0);
  CHECK_COMPLEX_NEAR(changed.sample.z1_scri[0], fine.sample.z1_scri[0], 0.0);
  CHECK(Kokkos::abs(changed.sample.z0_scri[5] - fine.sample.z0_scri[5]) >
        0.01);
  bool rejected = false;
  std::string reason;
  try {
    // This changes only mode index 1, theta index 2. The old scalar
    // certificate compared only mode 0, theta 0 and incorrectly accepted it.
    static_cast<void>(manufactured(97, 0.73, true, &certificate, true));
  } catch (const std::runtime_error& error) {
    rejected = true;
    reason = error.what();
  }
  CHECK(rejected);
  CHECK(reason == "plus2 endpoint coefficients do not match qualified data");
}

TEST_CASE("plus2 D10-5 peeling initialization is endpoint fourth order") {
  for (const double spin : {0.0, 0.73}) {
    const auto coarse = manufactured(25, spin, false);
    const auto medium = manufactured(49, spin, false);
    const auto fine = manufactured(97, spin, false);
    const auto certificate = teuk::qualify_plus2_peeling_convergence(
        coarse.sample, medium.sample, fine.sample,
        spin == 0.0 ? "manufactured-Schwarzschild-peeling-v1"
                    : "manufactured-rotating-Kerr-peeling-v1",
        4.0);
    CHECK(certificate.minimum_observed_order() >= 4.0);
    const auto initialized_fine =
        manufactured(97, spin, true, &certificate);
    std::cout << "plus2 peeling "
              << (spin == 0.0 ? "Schwarzschild" : "rotating-Kerr")
              << " minimum order "
              << certificate.minimum_observed_order() << " fine error "
              << initialized_fine.maximum_error << '\n';
    CHECK(initialized_fine.maximum_error < 1.0e-8);
  }
}
