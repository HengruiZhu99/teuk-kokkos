#include <algorithm>
#include <cmath>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular_source.hpp"

namespace source = teuk::angular_source;

namespace {

void fill_mode(source::HostModalField& field, const int m,
               const double scale) {
  auto& modal = field.mode(m);
  for (std::size_t ell = 0; ell < modal.size(); ++ell) {
    modal[ell] = teuk::Complex(scale * (1.0 + static_cast<double>(ell)),
                               scale * (0.25 - 0.5 * ell));
  }
}

void check_projection_near(
    const source::HostMultimodeProductProjection& actual,
    const source::HostMultimodeProductProjection& expected,
    const double tolerance) {
  CHECK(actual.targets.size() == expected.targets.size());
  CHECK(actual.pair_contributions.size() ==
        expected.pair_contributions.size());
  for (std::size_t pair = 0; pair < actual.pair_contributions.size(); ++pair) {
    CHECK(actual.pair_contributions[pair].pair ==
          expected.pair_contributions[pair].pair);
    for (std::size_t ell = 0;
         ell < actual.pair_contributions[pair].modal.size(); ++ell) {
      CHECK_COMPLEX_NEAR(actual.pair_contributions[pair].modal[ell],
                         expected.pair_contributions[pair].modal[ell],
                         tolerance);
    }
  }
  for (std::size_t target = 0; target < actual.targets.size(); ++target) {
    CHECK(actual.targets[target].target == expected.targets[target].target);
    for (std::size_t ell = 0; ell < actual.targets[target].modal.size();
         ++ell) {
      CHECK_COMPLEX_NEAR(actual.targets[target].modal[ell],
                         expected.targets[target].modal[ell], tolerance);
    }
  }
}

}  // namespace

TEST_CASE("multimode sharp field uses negative-mode lookup before conjugation") {
  const teuk::ModeRegistry registry({-2, -1, 0, 1, 2});
  source::HostModalField field(registry, -1, 4);
  fill_mode(field, -2, 0.7);
  fill_mode(field, 2, -0.2);

  const auto sharp_plus_two = field.sharp_mode(2);
  const auto wrong_same_mode = field.mode(2);
  CHECK(Kokkos::abs(sharp_plus_two[0] - Kokkos::conj(wrong_same_mode[0])) >
        1e-3);

  const int node_count = 8;
  const teuk::angular::SpinWeightedTransform original(-1, -2, 4,
                                                      node_count);
  const teuk::angular::SpinWeightedTransform sharp(+1, +2, 4, node_count);
  const auto original_nodal = original.synthesize(field.mode(-2));
  const auto sharp_nodal = sharp.synthesize(sharp_plus_two);
  for (std::size_t node = 0; node < original_nodal.size(); ++node) {
    CHECK_COMPLEX_NEAR(sharp_nodal[node], Kokkos::conj(original_nodal[node]),
                       3e-13);
  }
}

TEST_CASE("signed multimode Gaunt and padded projections agree pair by pair") {
  const teuk::ModeRegistry registry({-2, -1, 0, 1, 2}, {-2, 0, 2});
  source::HostModalField left(registry, -1, 4);
  source::HostModalField right(registry, -1, 4);
  fill_mode(left, -1, 0.3);
  fill_mode(left, 1, -0.4);
  fill_mode(right, -1, 0.5);
  fill_mode(right, 1, 0.2);

  const auto exact = source::project_product(
      registry, left, right, 5, source::HostProductMethod::ExactGaunt);
  const auto padded = source::project_product(
      registry, left, right, 5,
      source::HostProductMethod::PaddedCollocation);
  check_projection_near(padded, exact, 3e-11);

  CHECK(exact.targets.size() == 3);
  CHECK(exact.targets[0].target == -2);
  CHECK(exact.targets[1].target == 0);
  CHECK(exact.targets[2].target == 2);
  CHECK(exact.pair_contributions.size() ==
        registry.ordered_pairs().size());
  for (std::size_t pair = 0; pair < exact.pair_contributions.size(); ++pair) {
    CHECK(exact.pair_contributions[pair].pair ==
          registry.ordered_pairs()[pair]);
  }
}

TEST_CASE("target filtering excludes signed daughters outside target registry") {
  const teuk::ModeRegistry registry({-2, -1, 0, 1, 2}, {0});
  source::HostModalField left(registry, 0, 3);
  source::HostModalField right(registry, 0, 3);
  fill_mode(left, 1, 0.4);
  fill_mode(right, 1, -0.6);  // The nonzero daughter is m=2, not target m=0.

  const auto projection = source::project_product(
      registry, left, right, 4, source::HostProductMethod::ExactGaunt);
  CHECK(projection.targets.size() == 1);
  CHECK(projection.targets[0].target == 0);
  for (const auto value : projection.targets[0].modal) {
    CHECK_COMPLEX_NEAR(value, teuk::Complex(0.0, 0.0), 1e-15);
  }
  for (const auto& pair : projection.pair_contributions) {
    CHECK(pair.pair.target == 0);
    CHECK(pair.pair.m1 + pair.pair.m2 == 0);
  }
}

TEST_CASE("inner source projection invokes point algebra with explicit sharp field") {
  const teuk::ModeRegistry registry({-1, 0, 1}, {0});
  constexpr int ell_max = 3;
  source::HostModalField F(registry, -2, ell_max);
  source::HostModalField G(registry, -1, ell_max);
  source::HostModalField H(registry, 0, ell_max);
  source::HostModalField Lambda(registry, -2, ell_max);
  source::HostModalField Pi(registry, -1, ell_max);
  source::HostModalField B(registry, -2, ell_max);
  source::HostModalField C(registry, -1, ell_max);
  source::HostModalField U(registry, 0, ell_max);
  fill_mode(F, -1, 0.45);
  fill_mode(Pi, -1, -0.35);

  source::HostModalField delta1_F(registry, -2, ell_max);
  source::HostModalField delta3_U(registry, 0, ell_max);
  source::HostModalField eth2_C(registry, 0, ell_max);
  source::HostModalField ethprime2_C_sharp(registry, 0, ell_max);
  source::HostModalField eth1_B(registry, -1, ell_max);
  source::HostModalField delta2_C(registry, -1, ell_max);
  source::HostModalField delta2_G(registry, -1, ell_max);
  source::HostModalField eth2_G(registry, 0, ell_max);
  source::HostModalField ethprime1_F(registry, -3, ell_max);
  source::HostModalField delta2_C_sharp(registry, 1, ell_max);
  source::HostModalField ethprime1_B_sharp(registry, 1, ell_max);

  const source::HostSourceFields fields{F, G, H, Lambda, Pi, B, C, U};
  const source::HostSourceDerivatives derivatives{
      delta1_F, delta3_U, eth2_C, ethprime2_C_sharp, eth1_B, delta2_C,
      delta2_G, eth2_G, ethprime1_F, delta2_C_sharp,
      ethprime1_B_sharp};
  const auto projected = source::project_inner_source(
      registry, fields, derivatives, teuk::KerrParameters{1.0, 0.7, 1.2},
      0.4, ell_max, 10);

  // With every field but F and Pi zero, corrected source term st04 is exactly
  // F1*Pi2_sharp. Compare the complete point-algebra wrapper with the separate
  // exact-Gaunt multimode product oracle.
  const auto Pi_sharp = Pi.sharp_field();
  const auto exact = source::project_product(
      registry, F, Pi_sharp, ell_max,
      source::HostProductMethod::ExactGaunt);
  CHECK(projected.targets.size() == 1);
  CHECK(projected.targets[0].target == 0);
  for (const auto value : projected.targets[0].D) {
    CHECK_COMPLEX_NEAR(value, teuk::Complex(0.0, 0.0), 2e-13);
  }
  for (std::size_t ell = 0; ell < projected.targets[0].T.size(); ++ell) {
    CHECK_COMPLEX_NEAR(projected.targets[0].T[ell],
                       exact.targets[0].modal[ell], 3e-11);
  }
  CHECK(projected.pair_contributions.size() == 3);
  CHECK(projected.pair_contributions[0].pair.m1 == -1);
  CHECK(projected.pair_contributions[0].pair.m2 == 1);
}
