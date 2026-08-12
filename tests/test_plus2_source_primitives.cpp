#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "teuk/jet.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_source_primitives.hpp"

namespace {

using C = std::complex<double>;
using KC = teuk::Complex;
using Fields = teuk::Plus2ReconstructionPrimitiveInputsT<KC>;
using Derivatives = teuk::Plus2PrimitiveDerivativesT<KC>;
using Primitives = teuk::Plus2SourcePrimitivesT<KC>;

int primitive_allocations = 0;

void count_primitive_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                const void*, std::uint64_t) {
  ++primitive_allocations;
}

C host(const KC& value) { return {value.real(), value.imag()}; }
KC device(const C& value) { return {value.real(), value.imag()}; }

KC random_complex(std::mt19937_64& generator) {
  std::uniform_real_distribution<double> distribution(-0.8, 0.8);
  return {distribution(generator), distribution(generator)};
}

Fields random_fields(std::mt19937_64& g) {
  return {random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g)};
}

Derivatives random_derivatives(std::mt19937_64& g) {
  return {random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g)};
}

std::map<std::string, C> flatten(const Primitives& p) {
  return {{"p0", host(p.Z0)},
          {"p1", host(p.Z1)},
          {"p2", host(p.H)},
          {"sig", host(p.Sig)},
          {"kap", host(p.Kap)},
          {"rh", host(p.Rh)},
          {"ta", host(p.Ta)},
          {"al", host(p.Al)},
          {"be", host(p.Be)},
          {"ep", host(p.Ep)},
          {"pi", host(p.Pi)},
          {"hll", host(p.V)},
          {"hlb", host(p.C)},
          {"hbb", host(p.B)}};
}

template <class Extract>
std::map<std::string, C> flatten_all(
    const teuk::Plus2SourcePrimitivesT<teuk::Jet1<KC>>& p,
    Extract extract) {
  return {{"p0", extract(p.Z0)},
          {"p1", extract(p.Z1)},
          {"p2", extract(p.H)},
          {"sig", extract(p.Sig)},
          {"kap", extract(p.Kap)},
          {"rh", extract(p.Rh)},
          {"ta", extract(p.Ta)},
          {"al", extract(p.Al)},
          {"be", extract(p.Be)},
          {"ep", extract(p.Ep)},
          {"pi", extract(p.Pi)},
          {"hll", extract(p.V)},
          {"hlb", extract(p.C)},
          {"hbb", extract(p.B)},
          {"Rhsharp", extract(p.Rhsharp)},
          {"Alsharp", extract(p.Alsharp)},
          {"Epsharp", extract(p.Epsharp)},
          {"Pisharp", extract(p.Pisharp)},
          {"PiResidual", extract(p.Pi_input_residual)},
          {"Psi0Leading", extract(p.psi0_leading)},
          {"Psi0Residual", extract(p.psi0_quotient_residual)},
          {"Psi1Leading", extract(p.psi1_leading)},
          {"Psi1Residual", extract(p.psi1_quotient_residual)}};
}

std::map<std::string, C> flatten_all(const Primitives& p) {
  return {{"p0", host(p.Z0)},
          {"p1", host(p.Z1)},
          {"p2", host(p.H)},
          {"sig", host(p.Sig)},
          {"kap", host(p.Kap)},
          {"rh", host(p.Rh)},
          {"ta", host(p.Ta)},
          {"al", host(p.Al)},
          {"be", host(p.Be)},
          {"ep", host(p.Ep)},
          {"pi", host(p.Pi)},
          {"hll", host(p.V)},
          {"hlb", host(p.C)},
          {"hbb", host(p.B)},
          {"Rhsharp", host(p.Rhsharp)},
          {"Alsharp", host(p.Alsharp)},
          {"Epsharp", host(p.Epsharp)},
          {"Pisharp", host(p.Pisharp)},
          {"PiResidual", host(p.Pi_input_residual)},
          {"Psi0Leading", host(p.psi0_leading)},
          {"Psi0Residual", host(p.psi0_quotient_residual)},
          {"Psi1Leading", host(p.psi1_leading)},
          {"Psi1Residual", host(p.psi1_quotient_residual)}};
}

void check_maps(const std::map<std::string, C>& actual,
                const std::map<std::string, C>& expected,
                const double tolerance) {
  CHECK(actual.size() == expected.size());
  for (const auto& [name, value] : expected) {
    CHECK(actual.contains(name));
    CHECK(std::abs(actual.at(name) - value) <= tolerance);
  }
}

struct HostOracle {
  std::map<std::string, C> manifest;
  C Rhsharp, Alsharp, Epsharp, Pisharp;
  C psi0_leading, psi1_leading;
};

HostOracle ordinary_np_oracle(
    const double r, const teuk::Plus2PrimitiveBackground& bk,
    const Fields& fk, const Derivatives& dk) {
  // Independent ordinary-NP route.  It reconstructs the physical metric,
  // directional derivatives, and all Appendix-B formulas before dividing by
  // their manifest powers.  It does not call the production primitive helper.
  const auto z = [](const KC& value) { return host(value); };
  const C mu0 = z(bk.kerr.mu0), rho0 = z(bk.kerr.rho0);
  const C eps0 = z(bk.kerr.epsilon0), pi0 = z(bk.kerr.pi0);
  const C tau0 = z(bk.kerr.tau0), alpha0 = z(bk.alpha0);
  const C beta0 = z(bk.beta0), mubar0 = std::conj(mu0);
  const C rhobar0 = std::conj(rho0), epsbar0 = std::conj(eps0);
  const C pibar0 = std::conj(pi0), taubar0 = std::conj(tau0);
  const C alphabar0 = std::conj(alpha0), betabar0 = std::conj(beta0);
  const C U = z(fk.U), Us = z(fk.Usharp), C0 = z(fk.C);
  const C Cs = z(fk.Csharp), B = z(fk.B), Bs = z(fk.Bsharp);
  const C V = U / mu0, Vs = Us / mubar0;
  const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;

  const C hll = r2 * V, hlbar = r2 * C0, hlm = r2 * Cs;
  const C hbb = r * B, hmm = r * Bs;
  const C mu = r * mu0, mubar = r * mubar0;
  const C rho = r * rho0, rhobar = r * rhobar0;
  const C eps = r2 * eps0, epsbar = r2 * epsbar0;
  const C pi = r2 * pi0, pibar = r2 * pibar0;
  const C tau = r2 * tau0, taubar = r2 * taubar0;
  const C alpha = r * alpha0, alphabar = r * alphabar0;
  const C beta = r * beta0, betabar = r * betabar0;

  const C Delta_hlm = r2 * z(dk.capital_delta2_Csharp);
  const C Delta_hlbar = r2 * z(dk.capital_delta2_C);
  const C Delta_hll = r2 * z(dk.capital_delta2_V);
  const C D_hmm = r2 * z(dk.thorn1_Bsharp) +
                  2.0 * r3 * (eps0 - epsbar0) * Bs;
  const C D_hlm = r3 * z(dk.thorn2_Csharp) + 2.0 * r4 * eps0 * Cs;
  const C delta_hll =
      r3 * z(dk.eth2_V) + 2.0 * r3 * (beta0 + alphabar0) * V;
  const C bardelta_hlm =
      r3 * z(dk.ethprime2_Csharp) + 2.0 * r3 * alpha0 * Cs;
  const C delta_hlbar =
      r3 * z(dk.eth2_C) + 2.0 * r3 * alphabar0 * C0;
  const C delta_hbb = r2 * z(dk.eth1_B) - 2.0 * r2 * beta0 * B +
                      2.0 * r2 * alphabar0 * B;
  const C bardelta_hmm =
      r2 * z(dk.ethprime1_Bsharp) + 2.0 * r2 * alpha0 * Bs -
      2.0 * r2 * betabar0 * Bs;

  const C sigma = 0.5 *
                      (D_hmm + 2.0 * (epsbar - eps) * hmm +
                       (rho - rhobar) * hmm) -
                  (tau + pibar) * hlm;
  const C kappa = D_hlm - (2.0 * eps + rhobar) * hlm -
                  0.5 * (delta_hll - 2.0 * alphabar * hll -
                         2.0 * beta * hll + (pibar + tau) * hll);
  const C rho1 = 0.5 * mu * hll +
                 0.5 * (bardelta_hlm - 2.0 * alpha * hlm - pi * hlm) -
                 0.5 * (delta_hlbar - 2.0 * alphabar * hlbar +
                        (pibar + 2.0 * tau) * hlbar);
  const C tau1 = 0.5 * (Delta_hlm + mu * hlm) - 0.5 * pi * hmm;
  const C pi1 = -0.5 * (Delta_hlbar + mubar * hlbar) - 0.5 * tau * hbb;
  const C pibar1 =
      -0.5 * (Delta_hlm + mu * hlm) - 0.5 * taubar * hmm;
  const C alpha1 =
      -0.25 * (Delta_hlbar + (-2.0 * mu + mubar) * hlbar) -
      0.25 *
          (delta_hbb + (-2.0 * alphabar + pibar + tau) * hbb);
  const C alphabar1 =
      -0.25 * (Delta_hlm + (-2.0 * mubar + mu) * hlm) -
      0.25 *
          (bardelta_hmm + (-2.0 * alpha + pi + taubar) * hmm);
  const C beta1 =
      -0.25 * (Delta_hlm + (mu + 2.0 * mubar) * hlm) +
      0.25 *
          (bardelta_hmm + (2.0 * betabar - pi - taubar) * hmm);
  const C epsilon1 =
      -0.25 * (Delta_hll + (-mu + mubar) * hll) -
      0.25 *
          (delta_hlbar + (-2.0 * alphabar + pibar + 2.0 * tau) * hlbar) +
      0.25 *
          (bardelta_hlm + (-2.0 * alpha - 3.0 * pi - 2.0 * taubar) * hlm);

  const C Delta_hll_sharp = r2 * z(dk.capital_delta2_Vsharp);
  const C hll_sharp = r2 * Vs;
  const C epsilonbar1 =
      -0.25 * (Delta_hll_sharp + (-mubar + mu) * r2 * Vs) -
      0.25 *
          (bardelta_hlm + (-2.0 * alpha + pi + 2.0 * taubar) * hlm) +
      0.25 *
          (delta_hlbar + (-2.0 * alphabar - 3.0 * pibar - 2.0 * tau) *
                             hlbar);
  const C rhobar1 =
      0.5 * mubar * hll_sharp +
      0.5 * (delta_hlbar - 2.0 * alphabar * hlbar - pibar * hlbar) -
      0.5 * (bardelta_hlm - 2.0 * alpha * hlm +
             (pi + 2.0 * taubar) * hlm);

  const C D_sigma = r3 * z(dk.thorn2_Sig) +
                    r4 * (3.0 * eps0 - epsbar0) * (sigma / r2);
  const C delta_kappa = r4 * z(dk.eth3_Kap) +
                        r4 * (3.0 * beta0 + alphabar0) * (kappa / r3);
  const C psi0 = D_sigma - (rho + rhobar + 3.0 * eps - epsbar) * sigma -
                 (delta_kappa - alphabar * kappa - 3.0 * beta * kappa +
                  pibar * kappa - tau * kappa);

  const C D_beta1 = r3 * z(dk.thorn2_Be) +
                    r4 * (eps0 - epsbar0) * (beta1 / r2);
  const C delta_epsilon1 =
      r3 * z(dk.eth2_Ep) +
      r3 * (beta0 + alphabar0) * (epsilon1 / r2);
  const C D1_beta0 = -0.5 * r3 * V * z(dk.capital_delta1_beta0);
  const C delta1_epsilon0 =
      -r4 * Cs * z(dk.capital_delta2_epsilon0) +
      0.5 * r4 * Bs * z(dk.bardelta2_epsilon0);
  const C psi1 =
      D_beta1 + D1_beta0 - rhobar * beta1 -
      rhobar1 * r * beta0 + epsbar * beta1 +
      epsilonbar1 * r * beta0 - delta_epsilon1 - delta1_epsilon0 +
      alphabar * epsilon1 + alphabar1 * r2 * eps0 - pibar * epsilon1 -
      pibar1 * r2 * eps0 - (alpha + pi) * sigma + mu * kappa;

  HostOracle oracle;
  oracle.manifest = {{"p0", psi0 / r5},
                     {"p1", psi1 / r4},
                     {"p2", z(fk.H)},
                     {"sig", sigma / r2},
                     {"kap", kappa / r3},
                     {"rh", rho1 / r3},
                     {"ta", tau1 / r2},
                     {"al", alpha1 / r2},
                     {"be", beta1 / r2},
                     {"ep", epsilon1 / r2},
                     {"pi", pi1 / r2},
                     {"hll", V},
                     {"hlb", C0},
                     {"hbb", B}};
  oracle.Rhsharp = rhobar1 / r3;
  oracle.Alsharp = alphabar1 / r2;
  oracle.Epsharp = epsilonbar1 / r2;
  oracle.Pisharp = pibar1 / r2;
  oracle.psi0_leading = z(dk.thorn2_Sig) -
                        (rho0 + rhobar0) * (sigma / r2) -
                        r * z(dk.eth3_Kap);
  oracle.psi1_leading =
      z(dk.thorn2_Be) - rhobar0 * (beta1 / r2) - z(dk.eth2_Ep) -
      beta0 * (epsilon1 / r2) - alpha0 * (sigma / r2) -
      0.5 * V * z(dk.capital_delta1_beta0) +
      (epsilonbar1 / r2) * beta0;
  return oracle;
}

void enforce_consistency(const double r,
                         const teuk::Plus2PrimitiveBackground& background,
                         Fields& fields, Derivatives& d) {
  // First pass obtains the derived fields entering the two peeling numerators.
  auto p = teuk::plus2_source_primitives(r, background, fields, d);
  fields.Pi -= p.Pi_input_residual;
  const KC rho_sum = background.kerr.rho0 +
                     Kokkos::conj(background.kerr.rho0);
  d.thorn2_Sig = rho_sum * p.Sig + r * d.eth3_Kap +
                 r * r * d.psi0_leading_combination_over_r2;
  d.thorn2_Be =
      Kokkos::conj(background.kerr.rho0) * p.Be + d.eth2_Ep +
      background.beta0 * p.Ep + background.alpha0 * p.Sig +
      0.5 * p.V * d.capital_delta1_beta0 -
      p.Epsharp * background.beta0 +
      r * d.psi1_leading_combination_over_r;
}

std::vector<std::string> csv_ids() {
  std::ifstream input(std::string(TEUK_SOURCE_DIR) +
                      "/PLUS2_SOURCE_INPUT_MANIFEST.csv");
  CHECK(input.good());
  std::string line;
  CHECK(static_cast<bool>(std::getline(input, line)));
  std::vector<std::string> ids;
  while (std::getline(input, line)) {
    ids.push_back(line.substr(0, line.find(',')));
  }
  return ids;
}

Fields scaled(const Fields& f, const double a) {
  return {a * f.U, a * f.Usharp, a * f.C, a * f.Csharp,
          a * f.B, a * f.Bsharp, a * f.H, a * f.Pi};
}

Derivatives scaled_perturbations(const Derivatives& d, const double a) {
  return {a * d.thorn1_Bsharp,
          a * d.thorn2_Csharp,
          a * d.eth2_V,
          a * d.ethprime2_Csharp,
          a * d.eth2_C,
          a * d.capital_delta2_Csharp,
          a * d.capital_delta2_C,
          a * d.capital_delta2_V,
          a * d.capital_delta2_Vsharp,
          a * d.eth1_B,
          a * d.ethprime1_Bsharp,
          a * d.thorn2_Sig,
          a * d.eth3_Kap,
          a * d.psi0_leading_combination_over_r2,
          a * d.thorn2_Be,
          a * d.eth2_Ep,
          d.capital_delta1_beta0,
          d.capital_delta2_epsilon0,
          d.bardelta2_epsilon0,
          a * d.psi1_leading_combination_over_r};
}

Fields shifted(const Fields& f, const Fields& df, const double e) {
  Fields out = scaled(df, e);
  out.U += f.U;
  out.Usharp += f.Usharp;
  out.C += f.C;
  out.Csharp += f.Csharp;
  out.B += f.B;
  out.Bsharp += f.Bsharp;
  out.H += f.H;
  out.Pi += f.Pi;
  return out;
}

Derivatives shifted(const Derivatives& d, const Derivatives& dd,
                    const double e) {
  Derivatives out = scaled_perturbations(dd, e);
  out.thorn1_Bsharp += d.thorn1_Bsharp;
  out.thorn2_Csharp += d.thorn2_Csharp;
  out.eth2_V += d.eth2_V;
  out.ethprime2_Csharp += d.ethprime2_Csharp;
  out.eth2_C += d.eth2_C;
  out.capital_delta2_Csharp += d.capital_delta2_Csharp;
  out.capital_delta2_C += d.capital_delta2_C;
  out.capital_delta2_V += d.capital_delta2_V;
  out.capital_delta2_Vsharp += d.capital_delta2_Vsharp;
  out.eth1_B += d.eth1_B;
  out.ethprime1_Bsharp += d.ethprime1_Bsharp;
  out.thorn2_Sig += d.thorn2_Sig;
  out.eth3_Kap += d.eth3_Kap;
  out.psi0_leading_combination_over_r2 +=
      d.psi0_leading_combination_over_r2;
  out.thorn2_Be += d.thorn2_Be;
  out.eth2_Ep += d.eth2_Ep;
  // These three slots are fixed background directional derivatives.
  out.capital_delta1_beta0 = d.capital_delta1_beta0;
  out.capital_delta2_epsilon0 = d.capital_delta2_epsilon0;
  out.bardelta2_epsilon0 = d.bardelta2_epsilon0;
  out.psi1_leading_combination_over_r +=
      d.psi1_leading_combination_over_r;
  return out;
}

using J = teuk::Jet1<KC>;

J jet(const KC& z, const KC& dz) { return {z, dz}; }

teuk::Plus2ReconstructionPrimitiveInputsT<J> make_jets(
    const Fields& f, const Fields& df) {
  return {jet(f.U, df.U), jet(f.Usharp, df.Usharp), jet(f.C, df.C),
          jet(f.Csharp, df.Csharp), jet(f.B, df.B),
          jet(f.Bsharp, df.Bsharp), jet(f.H, df.H), jet(f.Pi, df.Pi)};
}

teuk::Plus2PrimitiveDerivativesT<J> make_jets(const Derivatives& d,
                                              const Derivatives& dd) {
  const KC zero{};
  return {jet(d.thorn1_Bsharp, dd.thorn1_Bsharp),
          jet(d.thorn2_Csharp, dd.thorn2_Csharp),
          jet(d.eth2_V, dd.eth2_V),
          jet(d.ethprime2_Csharp, dd.ethprime2_Csharp),
          jet(d.eth2_C, dd.eth2_C),
          jet(d.capital_delta2_Csharp, dd.capital_delta2_Csharp),
          jet(d.capital_delta2_C, dd.capital_delta2_C),
          jet(d.capital_delta2_V, dd.capital_delta2_V),
          jet(d.capital_delta2_Vsharp, dd.capital_delta2_Vsharp),
          jet(d.eth1_B, dd.eth1_B),
          jet(d.ethprime1_Bsharp, dd.ethprime1_Bsharp),
          jet(d.thorn2_Sig, dd.thorn2_Sig),
          jet(d.eth3_Kap, dd.eth3_Kap),
          jet(d.psi0_leading_combination_over_r2,
              dd.psi0_leading_combination_over_r2),
          jet(d.thorn2_Be, dd.thorn2_Be),
          jet(d.eth2_Ep, dd.eth2_Ep),
          jet(d.capital_delta1_beta0, zero),
          jet(d.capital_delta2_epsilon0, zero),
          jet(d.bardelta2_epsilon0, zero),
          jet(d.psi1_leading_combination_over_r,
              dd.psi1_leading_combination_over_r)};
}

bool finite(const C& z) {
  return std::isfinite(z.real()) && std::isfinite(z.imag());
}

TEST_CASE("spin plus2 primitive rows exactly reconcile with manifest") {
  const auto ids = csv_ids();
  const std::vector<std::string> expected = {
      "p0", "p1", "p2", "sig", "kap", "rh", "ta",
      "al", "be", "ep", "pi", "hll", "hlb", "hbb"};
  CHECK(ids == expected);
  std::mt19937_64 g(14);
  auto f = random_fields(g);
  auto d = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.71, 1.3}, 0.28, -0.33, std::sqrt(1.0 - 0.33 * 0.33));
  enforce_consistency(0.28, bg, f, d);
  CHECK(flatten(teuk::plus2_source_primitives(0.28, bg, f, d)).size() ==
        ids.size());
}

TEST_CASE("spin plus2 primitive background matches primary Kerr formulas") {
  for (const double r : {0.0, 0.17, 0.41}) {
    const double cosine = -0.36;
    const double sine = std::sqrt(1.0 - cosine * cosine);
    const teuk::KerrParameters p{1.07, 0.93, 1.31};
    const auto actual =
        teuk::plus2_primitive_background(p, r, cosine, sine);
    const C i(0.0, 1.0);
    const double l2 = p.compactification_length *
                      p.compactification_length;
    const C alpha0 = (cosine / sine) /
                     (2.0 * std::sqrt(2.0) *
                      (l2 + i * p.spin * r * cosine));
    const C beta0 =
        (-l2 * cosine / sine +
         i * p.spin * r * sine * (1.0 / (sine * sine) + 1.0)) /
        (2.0 * std::sqrt(2.0) *
         std::pow(l2 - i * p.spin * r * cosine, 2));
    CHECK(std::abs(host(actual.alpha0) - alpha0) < 2.0e-16);
    CHECK(std::abs(host(actual.beta0) - beta0) < 2.0e-16);
  }
}

TEST_CASE("spin plus2 primitive pi row preserves reconstructed input") {
  std::mt19937_64 g(1202);
  auto f = random_fields(g);
  auto d = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.72, 1.2}, 0.23, 0.2, std::sqrt(0.96));
  const KC supplied_pi = f.Pi;
  const auto p = teuk::plus2_source_primitives(0.23, bg, f, d);
  CHECK_COMPLEX_NEAR(p.Pi, supplied_pi, 0.0);
  CHECK(Kokkos::abs(p.Pi_input_residual) > 1.0e-6);
}

TEST_CASE("spin plus2 primitives match independent ordinary NP oracle") {
  for (const std::uint64_t seed : {23051932ULL, 2010162ULL, 9181981ULL}) {
    std::mt19937_64 g(seed);
    for (int point = 0; point < 4; ++point) {
      const double r = 0.08 + 0.09 * point;
      const double cosine = -0.61 + 0.37 * point;
      const teuk::KerrParameters parameters{1.0 + 0.03 * point,
                                            -0.83 + 0.41 * point,
                                            1.1 + 0.07 * point};
      const auto bg = teuk::plus2_primitive_background(
          parameters, r, cosine, std::sqrt(1.0 - cosine * cosine));
      auto f = random_fields(g);
      auto d = random_derivatives(g);
      enforce_consistency(r, bg, f, d);
      const auto actual = teuk::plus2_source_primitives(r, bg, f, d);
      const auto oracle = ordinary_np_oracle(r, bg, f, d);
      check_maps(flatten(actual), oracle.manifest, 5.0e-12);
      CHECK(std::abs(host(actual.Rhsharp) - oracle.Rhsharp) < 5.0e-12);
      CHECK(std::abs(host(actual.Alsharp) - oracle.Alsharp) < 5.0e-12);
      CHECK(std::abs(host(actual.Epsharp) - oracle.Epsharp) < 5.0e-12);
      CHECK(std::abs(host(actual.Pisharp) - oracle.Pisharp) < 5.0e-12);
      CHECK(std::abs(host(actual.psi0_leading) - oracle.psi0_leading) <
            5.0e-12);
      CHECK(std::abs(host(actual.psi1_leading) - oracle.psi1_leading) <
            5.0e-12);
      CHECK(Kokkos::abs(actual.Pi_input_residual) < 2.0e-14);
      CHECK(Kokkos::abs(actual.psi0_quotient_residual) < 2.0e-14);
      CHECK(Kokkos::abs(actual.psi1_quotient_residual) < 2.0e-14);
    }
  }
}

TEST_CASE("spin plus2 primitive sharp fields use signed partner mode") {
  const teuk::ModeRegistry registry({-2, 2});
  const std::array<KC, 2> values = {KC(0.3, -0.7), KC(-0.8, 0.2)};
  const std::size_t index = registry.index(2);
  const KC correct = Kokkos::conj(values[registry.sharp_index(2)]);
  const KC wrong = Kokkos::conj(values[index]);
  CHECK(Kokkos::abs(correct - wrong) > 0.1);
  Fields f{};
  Derivatives d{};
  f.U = KC(0.2, 0.1);
  f.Usharp = correct;
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.6, 1.2}, 0.25, 0.1, std::sqrt(0.99));
  const auto selected = teuk::plus2_source_primitives(0.25, bg, f, d);
  f.Usharp = wrong;
  const auto same_mode = teuk::plus2_source_primitives(0.25, bg, f, d);
  CHECK_COMPLEX_NEAR(selected.Rhsharp, 0.5 * correct, 1.0e-15);
  CHECK(Kokkos::abs(selected.Rhsharp - same_mode.Rhsharp) > 0.05);
}

TEST_CASE("spin plus2 primitive construction is linear in perturbation") {
  std::mt19937_64 g(314159);
  auto f = random_fields(g);
  auto d = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.88, 1.4}, 0.31, -0.24, std::sqrt(1.0 - 0.24 * 0.24));
  enforce_consistency(0.31, bg, f, d);
  constexpr double a = -2.75;
  auto af = scaled(f, a);
  auto ad = scaled_perturbations(d, a);
  const auto base = flatten_all(teuk::plus2_source_primitives(0.31, bg, f, d));
  const auto changed =
      flatten_all(teuk::plus2_source_primitives(0.31, bg, af, ad));
  for (const auto& [name, value] : base) {
    CHECK(std::abs(changed.at(name) - a * value) < 4.0e-12);
  }
}

TEST_CASE("spin plus2 primitive Jet tangent matches finite difference") {
  std::mt19937_64 g(271828);
  auto f = random_fields(g), df = random_fields(g);
  auto d = random_derivatives(g), dd = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, -0.67, 1.25}, 0.27, 0.29, std::sqrt(1.0 - 0.29 * 0.29));
  enforce_consistency(0.27, bg, f, d);
  // Pi and peeling quotient consistency are audit contracts, not required for
  // the directional algebra, so the arbitrary tangent may vary them.
  const auto analytic = flatten_all(
      teuk::plus2_source_primitives(0.27, bg, make_jets(f, df),
                                    make_jets(d, dd)),
      [](const J& z) { return host(z.dt); });
  constexpr double e = 2.0e-6;
  const auto plus = flatten_all(teuk::plus2_source_primitives(
      0.27, bg, shifted(f, df, e), shifted(d, dd, e)));
  const auto minus = flatten_all(teuk::plus2_source_primitives(
      0.27, bg, shifted(f, df, -e), shifted(d, dd, -e)));
  for (const auto& [name, tangent] : analytic) {
    const C difference = (plus.at(name) - minus.at(name)) / (2.0 * e);
    CHECK(std::abs(tangent - difference) < 2.0e-9);
  }
}

TEST_CASE("spin plus2 primitive endpoint rescalings remain finite") {
  std::mt19937_64 g(999);
  const teuk::KerrParameters parameters{1.0, 0.999, 1.2};
  const double boyer_lindquist_horizon =
      parameters.mass +
      std::sqrt(parameters.mass * parameters.mass -
                parameters.spin * parameters.spin);
  const double horizon = parameters.compactification_length *
                         parameters.compactification_length /
                         boyer_lindquist_horizon;
  for (const double r : {0.0, horizon}) {
    auto f = random_fields(g);
    auto d = random_derivatives(g);
    const auto bg = teuk::plus2_primitive_background(
        parameters, r, -0.37, std::sqrt(1.0 - 0.37 * 0.37));
    enforce_consistency(r, bg, f, d);
    const auto values = flatten_all(teuk::plus2_source_primitives(r, bg, f, d));
    for (const auto& [name, value] : values) {
      (void)name;
      CHECK(finite(value));
    }
    CHECK(std::abs(values.at("PiResidual")) < 2.0e-14);
    CHECK(std::abs(values.at("Psi0Residual")) < 2.0e-14);
    CHECK(std::abs(values.at("Psi1Residual")) < 2.0e-14);
    if (r == 0.0) {
      CHECK_COMPLEX_NEAR(r * r * device(values.at("sig")), KC{}, 0.0);
      CHECK_COMPLEX_NEAR(r * r * r * device(values.at("kap")), KC{}, 0.0);
      CHECK_COMPLEX_NEAR(std::pow(r, 5) * device(values.at("p0")), KC{}, 0.0);
      CHECK_COMPLEX_NEAR(std::pow(r, 4) * device(values.at("p1")), KC{}, 0.0);
    }
  }
}

TEST_CASE("spin plus2 primitive construction has host device parity") {
  std::mt19937_64 g(8675309);
  auto f = random_fields(g);
  auto d = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.91, 1.1}, 0.36, 0.42, std::sqrt(1.0 - 0.42 * 0.42));
  enforce_consistency(0.36, bg, f, d);
  const Primitives expected = teuk::plus2_source_primitives(0.36, bg, f, d);
  Kokkos::View<Primitives*> result("plus2_primitive_device", 1);
  Kokkos::parallel_for("plus2_primitive_device_parity", 1,
                       KOKKOS_LAMBDA(const int) {
                         result(0) = teuk::plus2_source_primitives(
                             0.36, bg, f, d);
                       });
  const auto copied = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          result);
  check_maps(flatten_all(copied(0)), flatten_all(expected), 1.0e-14);
}

TEST_CASE("spin plus2 primitive point evaluator allocates nothing") {
  std::mt19937_64 g(424242);
  auto f = random_fields(g);
  auto d = random_derivatives(g);
  const auto bg = teuk::plus2_primitive_background(
      {1.0, 0.52, 1.25}, 0.22, 0.1, std::sqrt(0.99));
  enforce_consistency(0.22, bg, f, d);
  KC sink{};
  primitive_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_primitive_allocation);
  for (int repeat = 0; repeat < 64; ++repeat) {
    sink += teuk::plus2_source_primitives(0.22, bg, f, d).Z0;
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(primitive_allocations == 0);
  CHECK(std::isfinite(sink.real()));
}

}  // namespace
