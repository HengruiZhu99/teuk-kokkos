#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "ripley_eq22_oracle.hpp"
#include "teuk/angular.hpp"
#include "teuk/boundary.hpp"
#include "teuk/plus2_companion_storage.hpp"
#include "teuk/plus2_field.hpp"
#include "teuk/teukolsky.hpp"

namespace {

int plus2_allocations = 0;
int plus2_launches = 0;

void count_plus2_allocation(Kokkos::Tools::SpaceHandle, const char*,
                            const void*, std::uint64_t) {
  ++plus2_allocations;
}

void count_plus2_launch(const char*, std::uint32_t, std::uint64_t*) {
  ++plus2_launches;
}

bool finite(const teuk::Complex value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void check_coefficients(
    const teuk::TeukolskyCoefficients& actual,
    const teuk::test::ripley_eq22::FirstOrderCoefficients& expected,
    const double tolerance) {
  CHECK_NEAR(actual.time, expected.time, tolerance);
  CHECK_NEAR(actual.radial_advection, expected.radial_advection, tolerance);
  CHECK_NEAR(actual.radial_principal, expected.radial_principal, tolerance);
  CHECK_COMPLEX_NEAR(
      actual.definition,
      teuk::Complex(expected.definition.real(), expected.definition.imag()),
      tolerance);
  CHECK_COMPLEX_NEAR(actual.q,
                     teuk::Complex(expected.q.real(), expected.q.imag()),
                     tolerance);
  CHECK_COMPLEX_NEAR(actual.psi,
                     teuk::Complex(expected.psi.real(), expected.psi.imag()),
                     tolerance);
}

}  // namespace

TEST_CASE("Ripley Eq. 21b plus field scaling has exact compact form") {
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.93;
  parameters.compactification_length = 1.7;
  const double radius = 0.41;
  const double cosine = -0.37;
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const teuk::Complex denominator(
      length2, -parameters.spin * radius * cosine);
  const teuk::Complex expected =
      std::pow(radius, 5) /
      (denominator * denominator * denominator * denominator);
  const auto scaling =
      teuk::plus2_code_tetrad_scaling(parameters, radius, cosine);
  CHECK_COMPLEX_NEAR(scaling, expected, 2.0e-18);

  const teuk::Complex regularized(0.7, -0.23);
  const auto psi0 = teuk::plus2_code_tetrad_psi0(
      regularized, parameters, radius, cosine);
  CHECK_COMPLEX_NEAR(
      teuk::plus2_regularized_from_code_tetrad_interior(
          psi0, parameters, radius, cosine),
      regularized, 2.0e-15);
}

TEST_CASE("plus field scaling is regular at scri and the future horizon") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.999;
  parameters.compactification_length = 2.3;
  const double cosine = 0.64;
  const double scri_coefficient = teuk::plus2_scri_scaling_coefficient(
      parameters.compactification_length);
  double previous_error = 1.0;
  for (const double radius : {1.0e-2, 2.0e-3, 4.0e-4}) {
    const auto scaling =
        teuk::plus2_code_tetrad_scaling(parameters, radius, cosine);
    const auto normalized = scaling / std::pow(radius, 5);
    const double error = Kokkos::abs(
        normalized - teuk::Complex(scri_coefficient, 0.0));
    CHECK(error < previous_error);
    previous_error = error;
  }
  CHECK(previous_error < 7.0e-5);

  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const auto horizon_scaling =
      teuk::plus2_code_tetrad_scaling(parameters, horizon, cosine);
  CHECK(finite(horizon_scaling));
  CHECK(Kokkos::abs(horizon_scaling) > 0.0);
}

TEST_CASE("plus field scaling helper executes in the active Kokkos space") {
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.81;
  parameters.compactification_length = 1.9;
  Kokkos::View<teuk::Complex*> output("plus2_scaling_device", 2);
  Kokkos::parallel_for(
      "plus2_scaling_kernel", 1, KOKKOS_LAMBDA(const int) {
        output(0) = teuk::plus2_code_tetrad_scaling(parameters, 0.37, -0.2);
        output(1) = teuk::plus2_code_tetrad_psi0(
            teuk::Complex(0.4, -0.7), parameters, 0.37, -0.2);
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  CHECK(finite(host(0)));
  CHECK_COMPLEX_NEAR(host(1), teuk::Complex(0.4, -0.7) * host(0),
                     2.0e-18);
}

TEST_CASE("generic spin plus2 coefficients equal independent Ripley Eq. 22") {
  struct Point {
    double mass;
    double spin;
    double length;
    int m;
    double horizon_fraction;
    double theta;
  };
  const std::array<Point, 6> points{{
      {1.0, 0.0, 1.4, 0, 0.17, 0.31},
      {1.0, 0.4, 1.7, -3, 0.23, 0.82},
      {1.3, -0.71, 2.2, 2, 0.49, 1.21},
      {0.8, 0.63, 1.1, -1, 0.76, 2.42},
      {1.0, 0.999, 2.0, 2, 0.91, 0.07},
      {2.0, -1.92, 0.9, -4, 0.58, 3.02},
  }};

  for (const auto& point : points) {
    teuk::TeukolskyParameters parameters;
    parameters.mass = point.mass;
    parameters.spin = point.spin;
    parameters.compactification_length = point.length;
    parameters.spin_weight = 2;
    parameters.azimuthal_mode = point.m;
    const double radius =
        point.horizon_fraction *
        teuk::compactified_outer_horizon_radius(parameters);
    const auto expected = teuk::test::ripley_eq22::coefficients(
        {point.mass, point.spin, point.length, point.m}, radius, point.theta);
    check_coefficients(
        teuk::teukolsky_coefficients(parameters, radius, point.theta),
        expected, 8.0e-13);
  }
}

TEST_CASE("spin plus2 Ripley coefficients have finite exact endpoint limits") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.2;
  parameters.spin = 1.19;
  parameters.compactification_length = 1.8;
  parameters.spin_weight = 2;
  parameters.azimuthal_mode = -3;
  const double theta = 0.73;
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const auto scri = teuk::teukolsky_coefficients(parameters, 0.0, theta);
  CHECK_NEAR(scri.time,
             16.0 * parameters.mass * parameters.mass -
                 parameters.spin * parameters.spin *
                     std::sin(theta) * std::sin(theta),
             3.0e-14);
  CHECK_NEAR(scri.radial_advection, length2, 0.0);
  CHECK_NEAR(scri.radial_principal, 0.0, 0.0);
  CHECK_COMPLEX_NEAR(
      scri.definition,
      teuk::Complex(-8.0 * parameters.mass,
                    2.0 * parameters.spin *
                        (parameters.azimuthal_mode + 2.0 * std::cos(theta))),
      3.0e-14);
  CHECK_COMPLEX_NEAR(scri.q, teuk::Complex(0.0, 0.0), 0.0);
  CHECK_COMPLEX_NEAR(scri.psi, teuk::Complex(0.0, 0.0), 0.0);

  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const auto horizon_coefficients =
      teuk::teukolsky_coefficients(parameters, horizon, theta);
  CHECK_NEAR(horizon_coefficients.radial_principal, 0.0, 2.0e-13);
  CHECK(std::isfinite(horizon_coefficients.time));
  CHECK(std::isfinite(horizon_coefficients.radial_advection));
  CHECK(finite(horizon_coefficients.definition));
  CHECK(finite(horizon_coefficients.q));
  CHECK(finite(horizon_coefficients.psi));

  check_coefficients(
      horizon_coefficients,
      teuk::test::ripley_eq22::coefficients(
          {parameters.mass, parameters.spin,
           parameters.compactification_length, parameters.azimuthal_mode},
          horizon, theta),
      8.0e-13);
}

TEST_CASE("spin plus2 angular action is Ripley lower after raise") {
  CHECK_NEAR(teuk::angular::spin_weighted_laplacian_eigenvalue(2, 2),
             0.0, 0.0);
  CHECK_NEAR(teuk::angular::spin_weighted_laplacian_eigenvalue(3, 2),
             -6.0, 0.0);
  CHECK_NEAR(teuk::angular::spin_weighted_laplacian_eigenvalue(4, 2),
             -14.0, 0.0);
  // At ell=s=2 the raising factor, and hence the composition, is exactly
  // zero; the intermediate spin-3 harmonic is absent.  For ell>=3 both
  // factors are represented explicitly by the angular library.
  CHECK_NEAR(teuk::angular::raising_factor(2, 2), 0.0, 0.0);
  for (int ell = 3; ell <= 8; ++ell) {
    const double explicit_lower_after_raise =
        teuk::angular::raising_factor(ell, 2) *
        teuk::angular::lowering_factor(ell, 3);
    CHECK_NEAR(
        teuk::angular::spin_weighted_laplacian_eigenvalue(ell, 2),
        explicit_lower_after_raise, 2.0e-14);
  }
}

TEST_CASE("spin plus2 coefficients have the explicit Schwarzschild limit") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.4;
  parameters.spin = 0.0;
  parameters.compactification_length = 2.1;
  parameters.spin_weight = 2;
  parameters.azimuthal_mode = 4;
  const double radius = 0.27;
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double x = parameters.mass * radius / length2;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, radius, 1.37);
  CHECK_NEAR(coefficients.time,
             16.0 * parameters.mass * parameters.mass * (1.0 + 2.0 * x),
             2.0e-14);
  CHECK_NEAR(coefficients.radial_advection,
             length2 - 8.0 * parameters.mass * parameters.mass * radius *
                           radius / length2,
             2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.definition,
      teuk::Complex(4.0 * parameters.mass * (-2.0 + 8.0 * x), 0.0),
      2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.q, teuk::Complex(2.0 * radius * (3.0 - 5.0 * x), 0.0),
      2.0e-14);
  CHECK_COMPLEX_NEAR(
      coefficients.psi, teuk::Complex(-6.0 * parameters.mass * radius /
                                          length2,
                                      0.0),
      2.0e-14);
}

TEST_CASE("disabled plus2 companion allocates and launches nothing") {
  // Positive controls ensure both callbacks are active before measuring the
  // disabled constructor.
  plus2_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_plus2_allocation);
  { Kokkos::View<double*> probe("plus2_allocation_probe", 1); }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(plus2_allocations > 0);

  plus2_launches = 0;
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_plus2_launch);
  Kokkos::parallel_for("plus2_launch_probe", 1,
                       KOKKOS_LAMBDA(const int) {});
  Kokkos::fence("observe plus2 launch probe");
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(plus2_launches > 0);

  plus2_allocations = 0;
  plus2_launches = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_plus2_allocation);
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_plus2_launch);
  const teuk::Plus2CompanionStorage disabled;
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);

  CHECK(!disabled.is_enabled());
  CHECK(disabled.mode_count() == 0);
  CHECK(disabled.radial_count() == 0);
  CHECK(disabled.theta_count() == 0);
  CHECK(disabled.value_count() == 0);
  CHECK(plus2_allocations == 0);
  CHECK(plus2_launches == 0);
}

TEST_CASE("enabled plus2 companion owns one triple and reusable RK scratch") {
  constexpr std::size_t modes = 3;
  constexpr std::size_t radial = 11;
  constexpr std::size_t theta = 7;
  plus2_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_plus2_allocation);
  auto storage = teuk::Plus2CompanionStorage::enabled(
      modes, radial, theta, "plus2_enabled_test");
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);

  CHECK(storage.is_enabled());
  CHECK(storage.mode_count() == modes);
  CHECK(storage.radial_count() == radial);
  CHECK(storage.theta_count() == theta);
  CHECK(storage.value_count() == modes * 3 * radial * theta);
  CHECK(storage.state().extent(0) == modes);
  CHECK(storage.state().extent(1) == 3);
  CHECK(storage.state().extent(2) == radial);
  CHECK(storage.state().extent(3) == theta);
  CHECK(storage.angular_laplacian().extent(0) == modes);
  CHECK(storage.radial_scratch().extent(1) == 5);
  CHECK(storage.flat_state().data() == storage.state().data());
  CHECK(storage.rk_workspace().size() == storage.value_count());
  CHECK(plus2_allocations > 0);
}

TEST_CASE("plus2 companion rejects invalid enabled extents") {
  bool rejected = false;
  try {
    auto invalid = teuk::Plus2CompanionStorage::enabled(2, 0, 5);
    static_cast<void>(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);

  bool disabled_access_rejected = false;
  try {
    const teuk::Plus2CompanionStorage disabled;
    static_cast<void>(disabled.state());
  } catch (const std::logic_error&) {
    disabled_access_rejected = true;
  }
  CHECK(disabled_access_rejected);
}
