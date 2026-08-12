#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <type_traits>

#include "teuk/plus2_linear_psi0.hpp"
#include "teuk/plus2_source_primitives.hpp"

namespace {

static_assert(!std::is_same_v<teuk::Plus2OrgMetricFields,
                              teuk::Plus2ReconstructionPrimitiveInputs>);

TEST_CASE("plus2 linear and primitive headers compose linear first") {
  constexpr double radius = 0.31;
  const auto background = teuk::plus2_primitive_background(
      {1.0, 0.82, 1.2}, radius, -0.27,
      Kokkos::sqrt(1.0 - 0.27 * 0.27));
  teuk::Plus2ReconstructionPrimitiveInputs input{};
  input.U = teuk::Complex(0.43, -0.18);
  input.Csharp = teuk::Complex(-0.22, 0.51);
  input.Bsharp = teuk::Complex(0.67, 0.29);

  const teuk::Plus2OrgMetricFields t0_metric =
      teuk::plus2_org_metric_from_reconstruction(
          radius, background.kerr.mu0, input.Bsharp, input.Csharp, input.U);
  CHECK_COMPLEX_NEAR(t0_metric.h_ll,
                     radius * radius * input.U / background.kerr.mu0,
                     1.0e-15);
  CHECK_COMPLEX_NEAR(t0_metric.h_lm, radius * radius * input.Csharp,
                     1.0e-15);
  CHECK_COMPLEX_NEAR(t0_metric.h_mm, radius * input.Bsharp, 1.0e-15);
}

TEST_CASE("plus2 evolved and source curvature variables compose exactly") {
  const teuk::Complex zplus(0.37, -0.21);
  for (const double radius : {0.0, 0.23, 0.71}) {
    const auto z0 = teuk::plus2_source_z0_from_evolved_zplus(
        zplus, radius, -0.34, 0.91, 1.7);
    const auto recovered = teuk::plus2_evolved_zplus_from_source_z0(
        z0, radius, -0.34, 0.91, 1.7);
    CHECK_COMPLEX_NEAR(recovered, zplus, 2.0e-15);
  }
  const double length = 1.7;
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const double length8 = length4 * length4;
  CHECK_COMPLEX_NEAR(
      teuk::plus2_source_z0_from_evolved_zplus(zplus, 0.0, 0.8, 0.999,
                                               length),
      zplus / length8, 2.0e-16);
}

}  // namespace
