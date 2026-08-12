#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "teuk/plus2_endpoint_extraction.hpp"

namespace {

using C = teuk::Complex;

struct EndpointResult {
  double error = 0.0;
  double residual = 0.0;
};

EndpointResult endpoint_result(const std::size_t points, const double spin,
                               const int signed_mode) {
  const double upper = 0.42;
  const double h = upper / static_cast<double>(points - 1);
  const double inverse_h = 1.0 / h;
  const double cosine = signed_mode > 0 ? 0.6 : -0.37;
  const C amplitude = signed_mode > 0 ? C(0.7, -0.21) : C(0.7, 0.21);
  const auto q0 = [&](const double radius) {
    const C dm(1.7 * 1.7, -spin * radius * cosine);
    return amplitude * Kokkos::exp(C(0.9 * radius, -0.23 * radius)) /
           (dm * dm * dm * dm);
  };
  const auto q1 = [&](const double radius) {
    return (C(0.3, -0.17) + amplitude) *
           Kokkos::exp(C(-0.4 * radius, 0.31 * radius));
  };
  const C residual_amplitude(0.13, -0.07);
  const C f0_constant = std::pow(h, 6) * residual_amplitude;
  const C f0_linear = std::pow(h, 5) * C(-0.09, 0.04);
  const C f1_constant = std::pow(h, 5) * C(0.06, 0.03);
  std::vector<C> f0(points), f1(points);
  for (std::size_t radial = 0; radial < points; ++radial) {
    const double radius = h * static_cast<double>(radial);
    f0[radial] =
        f0_constant + radius * f0_linear + radius * radius * q0(radius);
    f1[radial] = f1_constant + radius * q1(radius);
  }
  const C extracted0 = teuk::plus2_extract_q0_at_scri(
      f0.data(), points, inverse_h);
  const C extracted1 = teuk::plus2_extract_q1_at_scri(
      f1.data(), points, inverse_h);
  const auto residuals = teuk::plus2_peeling_residuals_at_scri(
      f0.data(), f1.data(), points, inverse_h);
  return {std::max(Kokkos::abs(extracted0 - q0(0.0)),
                   Kokkos::abs(extracted1 - q1(0.0))),
          std::max({Kokkos::abs(residuals.f0_constant),
                    Kokkos::abs(residuals.f0_linear),
                    Kokkos::abs(residuals.f1_constant)})};
}

}  // namespace

TEST_CASE("plus2 constrained endpoint weights have exact defining moments") {
  for (std::size_t power = 0; power < 6; ++power) {
    double moment = 0.0;
    double absolute_sum = 0.0;
    for (std::size_t node = 0; node < teuk::plus2_q0_endpoint_nodes; ++node) {
      const double term = teuk::plus2_q0_endpoint_weights[node] *
                          std::pow(static_cast<double>(node + 1),
                                   static_cast<int>(power));
      moment += term;
      absolute_sum += std::abs(term);
    }
    const double expected = power == 2 ? 1.0 : 0.0;
    CHECK(std::abs(moment - expected) <=
          8.0 * std::numeric_limits<double>::epsilon() * absolute_sum);
  }
  for (std::size_t power = 0; power < 5; ++power) {
    double moment = 0.0;
    double absolute_sum = 0.0;
    for (std::size_t node = 0; node < teuk::plus2_q1_endpoint_nodes; ++node) {
      const double term = teuk::plus2_q1_endpoint_weights[node] *
                          std::pow(static_cast<double>(node + 1),
                                   static_cast<int>(power));
      moment += term;
      absolute_sum += std::abs(term);
    }
    const double expected = power == 1 ? 1.0 : 0.0;
    CHECK(std::abs(moment - expected) <=
          8.0 * std::numeric_limits<double>::epsilon() * absolute_sum);
  }
  CHECK_NEAR(teuk::plus2_q0_endpoint_l1, 280.0 / 3.0, 0.0);
  CHECK_NEAR(teuk::plus2_q0_endpoint_l2_squared, 613067.0 / 288.0, 0.0);
  CHECK_NEAR(teuk::plus2_q0_endpoint_linf, 31.0, 0.0);
  CHECK_NEAR(teuk::plus2_q1_endpoint_l1, 56.0, 0.0);
  CHECK_NEAR(teuk::plus2_q1_endpoint_l2_squared, 60995.0 / 72.0, 0.0);
  CHECK_NEAR(teuk::plus2_q1_endpoint_linf, 19.5, 0.0);
}

TEST_CASE("plus2 constrained endpoint extraction is fourth order without hiding peeling") {
  for (const double spin : {0.0, 0.73, -0.73, 0.999}) {
    for (const int mode : {-2, 2}) {
      const auto n9 = endpoint_result(9, spin, mode);
      const auto n17 = endpoint_result(17, spin, mode);
      const auto n33 = endpoint_result(33, spin, mode);
      const auto n65 = endpoint_result(65, spin, mode);
      CHECK(n9.error / n17.error > 15.0);
      CHECK(n17.error / n33.error > 15.0);
      CHECK(n9.residual / n17.residual > 15.0);
      CHECK(n17.residual / n33.residual > 15.0);
      // N=65 is recorded only as a conditioning probe.  It is deliberately
      // excluded from promotion because the complete Route-B h4 tower is
      // already binary64-red there; this local stencil must not erase that
      // independent blocker.
      CHECK(std::isfinite(n65.error));
      CHECK(std::isfinite(n65.residual));
    }
  }
}

TEST_CASE("plus2 constrained endpoint extraction has host device parity") {
  constexpr std::size_t points = 9;
  Kokkos::View<C*> f0("endpoint_f0", points);
  Kokkos::View<C*> f1("endpoint_f1", points);
  Kokkos::View<C*> result("endpoint_result", 2);
  auto h0 = Kokkos::create_mirror_view(f0);
  auto h1 = Kokkos::create_mirror_view(f1);
  for (std::size_t i = 0; i < points; ++i) {
    const double radius = 0.05 * static_cast<double>(i);
    h0(i) = C(0.02, -0.01) + radius * C(-0.03, 0.02) +
            radius * radius * Kokkos::exp(C(radius, -0.2 * radius));
    h1(i) = C(-0.01, 0.005) +
            radius * Kokkos::exp(C(-0.4 * radius, 0.1 * radius));
  }
  Kokkos::deep_copy(f0, h0);
  Kokkos::deep_copy(f1, h1);
  Kokkos::parallel_for(
      "plus2_endpoint_extraction_device", 1, KOKKOS_LAMBDA(const int) {
        result(0) =
            teuk::plus2_extract_q0_at_scri(f0.data(), points, 20.0);
        result(1) =
            teuk::plus2_extract_q1_at_scri(f1.data(), points, 20.0);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        result);
  CHECK_COMPLEX_NEAR(host(0),
                     teuk::plus2_extract_q0_at_scri(h0.data(), points, 20.0),
                     0.0);
  CHECK_COMPLEX_NEAR(host(1),
                     teuk::plus2_extract_q1_at_scri(h1.data(), points, 20.0),
                     0.0);
}
