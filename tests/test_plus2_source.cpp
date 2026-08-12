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
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/jet.hpp"
#include "teuk/ghp.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_source.hpp"

namespace {

using C = std::complex<double>;
using KC = teuk::Complex;
using Fields = teuk::Plus2OrderedPairFieldsT<KC>;
using Derivatives = teuk::Plus2OrderedPairDerivativesT<KC>;
using Source = teuk::Plus2OrderedPairSourceT<KC>;
using OuterDerivatives = teuk::Plus2OuterDerivativesT<KC>;
using Outer = teuk::Plus2OuterSourceTermsT<KC>;
using RegularizedOuterDerivatives =
    teuk::Plus2RegularizedOuterDerivativesT<KC>;
using OuterR7 = teuk::Plus2OuterSourceOverR7TermsT<KC>;

int plus2_source_allocations = 0;

void count_plus2_source_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                   const void*, std::uint64_t) {
  ++plus2_source_allocations;
}

C host(const KC& value) { return {value.real(), value.imag()}; }
KC device(const C& value) { return {value.real(), value.imag()}; }

struct HostBackground {
  C mu0, tau0, pi0, psi20, rho0, epsilon0;
};

HostBackground host(const teuk::KerrBackgroundPoint& b) {
  return {host(b.mu0), host(b.tau0), host(b.pi0), host(b.psi20),
          host(b.rho0), host(b.epsilon0)};
}

C host_regularized_radial_j(
    const C J, const C JT, const C JR, const int mode, const double radius,
    const double cosine, const teuk::KerrParameters& p,
    const HostBackground& bg) {
  const C ii(0.0, 1.0);
  const double l2 = p.compactification_length * p.compactification_length;
  const double l4 = l2 * l2;
  const double D = l4 + p.spin * p.spin * radius * radius * cosine * cosine;
  const double E = l2 - 2.0 * p.mass * radius +
                   p.spin * p.spin * radius * radius / l2;
  const double A = 2.0 * p.mass *
                   (2.0 * p.mass - p.spin * p.spin * radius / l2) / D;
  const double b = -0.5 * E / D;
  const C cancellation =
      ii * p.spin * cosine * E *
      (3.0 * l2 + 5.0 * ii * p.spin * radius * cosine) / (2.0 * D * D);
  const C coefficient =
      cancellation + ii * p.spin * static_cast<double>(mode) / D -
                        3.0 * bg.epsilon0 + std::conj(bg.epsilon0);
  return A * JT + b * JR + coefficient * J;
}

C host_source_over_r7(const double radius, const HostBackground& bg,
                      const C K, const C Q, const C regularized_radial_j,
                      const C eth6_k) {
  return regularized_radial_j + eth6_k - 4.0 * radius * bg.tau0 * K +
         radius * std::conj(bg.pi0) * K - 3.0 * bg.psi20 * Q;
}

template <class Scalar, class Extract>
std::map<std::string, C> flatten_pair(
    const teuk::Plus2OrderedPairSourceT<Scalar>& s, Extract extract) {
  return {
      {"c01", extract(s.C12.c01)}, {"c02", extract(s.C12.c02)},
      {"c03", extract(s.C12.c03)}, {"c04", extract(s.C12.c04)},
      {"c05", extract(s.C12.c05)}, {"c06", extract(s.C12.c06)},
      {"c07", extract(s.C12.c07)}, {"c08", extract(s.C12.c08)},
      {"b01", extract(s.B12.b01)}, {"b02", extract(s.B12.b02)},
      {"b03", extract(s.B12.b03)}, {"b04", extract(s.B12.b04)},
      {"b05", extract(s.B12.b05)}, {"b06", extract(s.B12.b06)},
      {"b07", extract(s.B12.b07)}, {"b08", extract(s.B12.b08)},
      {"d01", extract(s.D12.d01)}, {"d02", extract(s.D12.d02)},
      {"d03", extract(s.D12.d03)}, {"d04", extract(s.D12.d04)},
      {"d05", extract(s.D12.d05)}, {"d06", extract(s.D12.d06)},
      {"d07", extract(s.D12.d07)}, {"d08", extract(s.D12.d08)},
      {"d09", extract(s.D12.d09)}, {"d10", extract(s.D12.d10)},
      {"er01", extract(s.Er12.er01)}, {"er02", extract(s.Er12.er02)},
      {"er03", extract(s.Er12.er03)}, {"er04", extract(s.Er12.er04)},
      {"er05", extract(s.Er12.er05)},
      {"et01", extract(s.Et12.et01)}, {"et02", extract(s.Et12.et02)},
      {"et03", extract(s.Et12.et03)}, {"et04", extract(s.Et12.et04)},
      {"et05", extract(s.Et12.et05)}, {"et06", extract(s.Et12.et06)},
      {"j01", extract(s.J12.j01)}, {"j02", extract(s.J12.j02)},
      {"k01", extract(s.K12.k01)}, {"k02", extract(s.K12.k02)},
      {"k03", extract(s.K12.k03)}, {"q01", extract(s.Q12.q01)},
      {"q02", extract(s.Q12.q02)},
  };
}

template <class Scalar, class Extract>
std::map<std::string, C> flatten_outer(
    const teuk::Plus2OuterSourceTermsT<Scalar>& s, Extract extract) {
  return {{"s01", extract(s.s01)}, {"s02", extract(s.s02)},
          {"s03", extract(s.s03)}, {"s04", extract(s.s04)},
          {"s05", extract(s.s05)}, {"s06", extract(s.s06)},
          {"s07", extract(s.s07)}};
}

void append(std::map<std::string, C>& destination,
            const std::map<std::string, C>& source) {
  destination.insert(source.begin(), source.end());
}

struct HostOracleResult {
  std::map<std::string, C> terms;
  C C12{}, B12{}, D12{}, Er12{}, Et12{}, J12{}, K12{}, Q12{};
};

HostOracleResult host_pair_oracle(const double r, const HostBackground& bg,
                                  const Fields& fk,
                                  const Derivatives& dk) {
  // Independent host transcription of PLUS2_SOURCE_TERM_LEDGER.csv.  It uses
  // std::complex and never calls the device-compatible implementation.
  const auto f = [&](const KC& z) { return host(z); };
  const C V = f(fk.V1), C1 = f(fk.C1), Cs = f(fk.Csharp1);
  const C B = f(fk.B1), Bs = f(fk.Bsharp1), Sig1 = f(fk.Sig1);
  const C Kap1 = f(fk.Kap1), Rh = f(fk.Rh1), Rhs = f(fk.Rhsharp1);
  const C Ep = f(fk.Ep1), Eps = f(fk.Epsharp1), Z0 = f(fk.Z0_2);
  const C Z1 = f(fk.Z1_2), H = f(fk.H2), Sig2 = f(fk.Sig2);
  const C Kap2 = f(fk.Kap2);
  const C mubar = std::conj(bg.mu0), pibar = std::conj(bg.pi0);
  const C taubar = std::conj(bg.tau0);
  const double r2 = r * r;
  HostOracleResult o;
  auto& x = o.terms;
  x["c01"] = -Cs * f(dk.capital_delta4_Z1_2);
  x["c02"] = 0.5 * Bs * f(dk.ethprime4_Z1_2);
  x["c03"] = -1.5 * f(dk.capital_delta2_Csharp1) * Z1;
  x["c04"] = -0.5 * f(dk.ethprime1_Bsharp1) * Z1;
  x["c05"] = -1.5 * r * bg.mu0 * Cs * Z1;
  x["c06"] = r * mubar * Cs * Z1;
  x["c07"] = 2.5 * r * bg.pi0 * Bs * Z1;
  x["c08"] = 0.5 * r * taubar * Bs * Z1;
  x["b01"] = -C1 * f(dk.capital_delta5_Z0_2);
  x["b02"] = 0.5 * B * f(dk.eth5_Z0_2);
  x["b03"] = 0.5 * f(dk.capital_delta2_C1) * Z0;
  x["b04"] = f(dk.eth1_B1) * Z0;
  x["b05"] = -2.0 * r * bg.mu0 * C1 * Z0;
  x["b06"] = 0.5 * r * mubar * C1 * Z0;
  x["b07"] = r * pibar * B * Z0;
  x["b08"] = 0.5 * r * bg.tau0 * B * Z0;
  x["d01"] = -0.5 * V * f(dk.capital_delta4_Z1_2);
  x["d02"] = 0.5 * f(dk.capital_delta2_V1) * Z1;
  x["d03"] = 2.5 * r * f(dk.eth2_C1) * Z1;
  x["d04"] = -2.5 * r * f(dk.ethprime2_Csharp1) * Z1;
  x["d05"] = -2.5 * r * bg.mu0 * V * Z1;
  x["d06"] = 0.5 * r * mubar * V * Z1;
  x["d07"] = 2.5 * r2 * pibar * C1 * Z1;
  x["d08"] = 5.0 * r2 * bg.tau0 * C1 * Z1;
  x["d09"] = 3.5 * r2 * bg.pi0 * Cs * Z1;
  x["d10"] = r2 * taubar * Cs * Z1;
  x["er01"] = -0.5 * V * f(dk.capital_delta2_Sig2);
  x["er02"] = -3.0 * Ep * Sig2;
  x["er03"] = Eps * Sig2;
  x["er04"] = -r * Rh * Sig2;
  x["er05"] = -r * Rhs * Sig2;
  x["et01"] = -Cs * f(dk.capital_delta3_Kap2);
  x["et02"] = 0.5 * Bs * f(dk.ethprime3_Kap2);
  x["et03"] = -0.5 * f(dk.ethprime1_Bsharp1) * Kap2;
  x["et04"] = r * mubar * Cs * Kap2;
  x["et05"] = 1.5 * r * bg.pi0 * Bs * Kap2;
  x["et06"] = 0.5 * r * taubar * Bs * Kap2;
  const auto sum = [&](const std::string& prefix, const int count) {
    C result{};
    for (int i = 1; i <= count; ++i) {
      std::ostringstream id;
      id << prefix;
      if (prefix == "d") {
        id << (i < 10 ? "0" : "") << i;
      } else {
        id << '0' << i;
      }
      result += x.at(id.str());
    }
    return result;
  };
  o.C12 = sum("c", 8);
  o.B12 = sum("b", 8);
  o.D12 = sum("d", 10);
  o.Er12 = sum("er", 5);
  o.Et12 = sum("et", 6);
  x["j01"] = r * o.C12;
  x["j02"] = 3.0 * Sig1 * H;
  o.J12 = x["j01"] + x["j02"];
  x["k01"] = r * o.B12;
  x["k02"] = -o.D12;
  x["k03"] = -3.0 * Kap1 * H;
  o.K12 = x["k01"] + x["k02"] + x["k03"];
  x["q01"] = o.Er12;
  x["q02"] = -r * o.Et12;
  o.Q12 = x["q01"] + x["q02"];
  return o;
}

std::map<std::string, C> host_outer_oracle(
    const double r, const HostBackground& bg, const C J, const C K,
    const C Q, const OuterDerivatives& dk) {
  const C thorn = host(dk.thorn5_J), eth = host(dk.eth6_K);
  return {{"s01", thorn},
          {"s02", -4.0 * bg.rho0 * J},
          {"s03", -std::conj(bg.rho0) * J},
          {"s04", r * eth},
          {"s05", -4.0 * r * r * bg.tau0 * K},
          {"s06", r * r * std::conj(bg.pi0) * K},
          {"s07", -3.0 * r * bg.psi20 * Q}};
}

KC random_complex(std::mt19937_64& generator) {
  std::uniform_real_distribution<double> distribution(-0.9, 0.9);
  return {distribution(generator), distribution(generator)};
}

Fields random_fields(std::mt19937_64& g) {
  return {random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g)};
}

Derivatives random_derivatives(std::mt19937_64& g) {
  return {random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g), random_complex(g),
          random_complex(g), random_complex(g)};
}

OuterDerivatives random_outer(std::mt19937_64& g) {
  return {random_complex(g), random_complex(g)};
}

void check_maps(const std::map<std::string, C>& actual,
                const std::map<std::string, C>& expected,
                const double tolerance) {
  CHECK(actual.size() == expected.size());
  for (const auto& [id, value] : expected) {
    CHECK(actual.contains(id));
    CHECK(std::abs(actual.at(id) - value) <= tolerance);
  }
}

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> result;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) result.push_back(field);
  return result;
}

std::map<std::string, std::string> load_ledger_ids() {
  std::ifstream input(std::string(TEUK_SOURCE_DIR) +
                      "/PLUS2_SOURCE_TERM_LEDGER.csv");
  CHECK(input.good());
  std::string line;
  CHECK(static_cast<bool>(std::getline(input, line)));
  std::map<std::string, std::string> result;
  while (std::getline(input, line)) {
    const auto columns = split_csv(line);
    CHECK(columns.size() == 11);
    CHECK(result.emplace(columns[0], columns[1]).second);
  }
  return result;
}

Fields scaled(const Fields& f, const double a) {
  return {a * f.V1,         a * f.C1,       a * f.Csharp1,
          a * f.B1,         a * f.Bsharp1,  a * f.Sig1,
          a * f.Kap1,       a * f.Rh1,      a * f.Rhsharp1,
          a * f.Ep1,        a * f.Epsharp1, a * f.Z0_2,
          a * f.Z1_2,       a * f.H2,       a * f.Sig2,
          a * f.Kap2};
}

Derivatives scaled(const Derivatives& d, const double a) {
  return {a * d.capital_delta4_Z1_2,
          a * d.ethprime4_Z1_2,
          a * d.capital_delta2_Csharp1,
          a * d.ethprime1_Bsharp1,
          a * d.capital_delta5_Z0_2,
          a * d.eth5_Z0_2,
          a * d.capital_delta2_C1,
          a * d.eth1_B1,
          a * d.capital_delta2_V1,
          a * d.eth2_C1,
          a * d.ethprime2_Csharp1,
          a * d.capital_delta2_Sig2,
          a * d.capital_delta3_Kap2,
          a * d.ethprime3_Kap2};
}

OuterDerivatives scaled(const OuterDerivatives& d, const double a) {
  return {a * d.thorn5_J, a * d.eth6_K};
}

template <class T>
teuk::Jet1<KC> jet(const T& value, const T& tangent) {
  return {value, tangent};
}

teuk::Plus2OrderedPairFieldsT<teuk::Jet1<KC>> make_jets(
    const Fields& f, const Fields& df) {
  return {jet(f.V1, df.V1), jet(f.C1, df.C1),
          jet(f.Csharp1, df.Csharp1), jet(f.B1, df.B1),
          jet(f.Bsharp1, df.Bsharp1), jet(f.Sig1, df.Sig1),
          jet(f.Kap1, df.Kap1), jet(f.Rh1, df.Rh1),
          jet(f.Rhsharp1, df.Rhsharp1), jet(f.Ep1, df.Ep1),
          jet(f.Epsharp1, df.Epsharp1), jet(f.Z0_2, df.Z0_2),
          jet(f.Z1_2, df.Z1_2), jet(f.H2, df.H2),
          jet(f.Sig2, df.Sig2), jet(f.Kap2, df.Kap2)};
}

teuk::Plus2OrderedPairDerivativesT<teuk::Jet1<KC>> make_jets(
    const Derivatives& d, const Derivatives& dd) {
  return {jet(d.capital_delta4_Z1_2, dd.capital_delta4_Z1_2),
          jet(d.ethprime4_Z1_2, dd.ethprime4_Z1_2),
          jet(d.capital_delta2_Csharp1, dd.capital_delta2_Csharp1),
          jet(d.ethprime1_Bsharp1, dd.ethprime1_Bsharp1),
          jet(d.capital_delta5_Z0_2, dd.capital_delta5_Z0_2),
          jet(d.eth5_Z0_2, dd.eth5_Z0_2),
          jet(d.capital_delta2_C1, dd.capital_delta2_C1),
          jet(d.eth1_B1, dd.eth1_B1),
          jet(d.capital_delta2_V1, dd.capital_delta2_V1),
          jet(d.eth2_C1, dd.eth2_C1),
          jet(d.ethprime2_Csharp1, dd.ethprime2_Csharp1),
          jet(d.capital_delta2_Sig2, dd.capital_delta2_Sig2),
          jet(d.capital_delta3_Kap2, dd.capital_delta3_Kap2),
          jet(d.ethprime3_Kap2, dd.ethprime3_Kap2)};
}

Fields shifted(const Fields& f, const Fields& df, const double e) {
  Fields result = scaled(df, e);
  result.V1 += f.V1;
  result.C1 += f.C1;
  result.Csharp1 += f.Csharp1;
  result.B1 += f.B1;
  result.Bsharp1 += f.Bsharp1;
  result.Sig1 += f.Sig1;
  result.Kap1 += f.Kap1;
  result.Rh1 += f.Rh1;
  result.Rhsharp1 += f.Rhsharp1;
  result.Ep1 += f.Ep1;
  result.Epsharp1 += f.Epsharp1;
  result.Z0_2 += f.Z0_2;
  result.Z1_2 += f.Z1_2;
  result.H2 += f.H2;
  result.Sig2 += f.Sig2;
  result.Kap2 += f.Kap2;
  return result;
}

Derivatives shifted(const Derivatives& d, const Derivatives& dd,
                    const double e) {
  Derivatives result = scaled(dd, e);
  result.capital_delta4_Z1_2 += d.capital_delta4_Z1_2;
  result.ethprime4_Z1_2 += d.ethprime4_Z1_2;
  result.capital_delta2_Csharp1 += d.capital_delta2_Csharp1;
  result.ethprime1_Bsharp1 += d.ethprime1_Bsharp1;
  result.capital_delta5_Z0_2 += d.capital_delta5_Z0_2;
  result.eth5_Z0_2 += d.eth5_Z0_2;
  result.capital_delta2_C1 += d.capital_delta2_C1;
  result.eth1_B1 += d.eth1_B1;
  result.capital_delta2_V1 += d.capital_delta2_V1;
  result.eth2_C1 += d.eth2_C1;
  result.ethprime2_Csharp1 += d.ethprime2_Csharp1;
  result.capital_delta2_Sig2 += d.capital_delta2_Sig2;
  result.capital_delta3_Kap2 += d.capital_delta3_Kap2;
  result.ethprime3_Kap2 += d.ethprime3_Kap2;
  return result;
}

OuterDerivatives shifted(const OuterDerivatives& d,
                         const OuterDerivatives& dd, const double e) {
  return {d.thorn5_J + e * dd.thorn5_J, d.eth6_K + e * dd.eth6_K};
}

TEST_CASE("spin plus2 source ledger has exact reflected term coverage") {
  const auto ledger = load_ledger_ids();
  CHECK(ledger.size() == teuk::plus2_compact_source_ledger_term_count);
  const std::map<std::string, int> expected_counts = {
      {"C12", 8}, {"B12", 8}, {"D12", 10}, {"Er12", 5},
      {"Et12", 6}, {"J12", 2}, {"K12", 3},  {"Q12", 2},
      {"S0", 7}};
  std::map<std::string, int> counts;
  for (const auto& [id, family] : ledger) {
    CHECK(!id.empty());
    ++counts[family];
  }
  CHECK(counts == expected_counts);

  std::mt19937_64 g(51);
  const Fields f = random_fields(g);
  const Derivatives d = random_derivatives(g);
  const OuterDerivatives od = random_outer(g);
  const auto bg = teuk::kerr_background_point({1.0, 0.73, 1.2}, 0.31,
                                               -0.27, std::sqrt(1.0 - 0.27 * 0.27));
  const auto pair = teuk::plus2_compact_ordered_pair_source(0.31, bg, f, d);
  const auto outer = teuk::plus2_compact_outer_source_over_r6(
      0.31, bg, pair.J12.total(), pair.K12.total(), pair.Q12.total(), od);
  auto reflected = flatten_pair(pair, [](const KC& z) { return host(z); });
  append(reflected, flatten_outer(outer, [](const KC& z) { return host(z); }));
  CHECK(reflected.size() == ledger.size());
  for (const auto& [id, family] : ledger) {
    (void)family;
    CHECK(reflected.contains(id));
    CHECK(std::abs(reflected.at(id)) > 1.0e-14);
  }
}

TEST_CASE("spin plus2 source agrees termwise with independent host oracle") {
  for (std::uint64_t seed : {230519332ULL, 201000162ULL, 9811019ULL,
                             511234567ULL}) {
    std::mt19937_64 g(seed);
    for (int point = 0; point < 4; ++point) {
      const double r = 0.04 + 0.11 * point;
      const double cosine = -0.67 + 0.39 * point;
      const teuk::KerrParameters parameters{1.0 + 0.05 * point,
                                            -0.81 + 0.37 * point,
                                            1.1 + 0.08 * point};
      const auto bg = teuk::kerr_background_point(
          parameters, r, cosine, std::sqrt(1.0 - cosine * cosine));
      const Fields f = random_fields(g);
      const Derivatives d = random_derivatives(g);
      const OuterDerivatives od = random_outer(g);
      const auto actual = teuk::plus2_compact_ordered_pair_source(r, bg, f, d);
      const auto expected = host_pair_oracle(r, host(bg), f, d);
      check_maps(flatten_pair(actual, [](const KC& z) { return host(z); }),
                 expected.terms, 2.0e-13);
      CHECK(std::abs(host(actual.C12.total()) - expected.C12) < 2.0e-13);
      CHECK(std::abs(host(actual.B12.total()) - expected.B12) < 2.0e-13);
      CHECK(std::abs(host(actual.D12.total()) - expected.D12) < 2.0e-13);
      CHECK(std::abs(host(actual.Er12.total()) - expected.Er12) < 2.0e-13);
      CHECK(std::abs(host(actual.Et12.total()) - expected.Et12) < 2.0e-13);
      CHECK(std::abs(host(actual.J12.total()) - expected.J12) < 2.0e-13);
      CHECK(std::abs(host(actual.K12.total()) - expected.K12) < 2.0e-13);
      CHECK(std::abs(host(actual.Q12.total()) - expected.Q12) < 2.0e-13);
      const auto outer = teuk::plus2_compact_outer_source_over_r6(
          r, bg, actual.J12.total(), actual.K12.total(), actual.Q12.total(), od);
      check_maps(flatten_outer(outer, [](const KC& z) { return host(z); }),
                 host_outer_oracle(r, host(bg), expected.J12, expected.K12,
                                   expected.Q12, od),
                 2.0e-13);
    }
  }
}

TEST_CASE("spin plus2 pair selection uses signed target and sharp partner") {
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-2, 2},
                                    {-4, 0, 4});
  const auto zero_range = registry.pair_range(0);
  CHECK(zero_range.second - zero_range.first == 2);
  const auto pair = registry.ordered_pairs().at(zero_range.first);
  const auto lookup = teuk::make_plus2_pair_lookup(registry, pair);
  CHECK(lookup.m1 + lookup.m2 == lookup.target);
  CHECK(lookup.target == 0);
  CHECK(registry.modes()[lookup.sharp1] == -lookup.m1);
  CHECK(registry.modes()[lookup.sharp2] == -lookup.m2);
  std::array<C, 5> values = {C(0.1, 0.2), C(0.3, 0.4), C(0.5, 0.6),
                             C(0.7, 0.8), C(0.9, 1.0)};
  const C correct = std::conj(values[lookup.sharp1]);
  const C wrong = std::conj(values[lookup.index1]);
  CHECK(std::abs(correct - wrong) > 0.1);
  Fields source_fields{};
  Derivatives source_derivatives{};
  source_fields.Csharp1 = device(correct);
  source_derivatives.capital_delta4_Z1_2 = KC(1.0, 0.0);
  const teuk::KerrBackgroundPoint zero_background{};
  const auto sharp_source = teuk::plus2_compact_ordered_pair_source(
      0.2, zero_background, source_fields, source_derivatives);
  source_fields.Csharp1 = device(wrong);
  const auto same_mode_conjugate_source =
      teuk::plus2_compact_ordered_pair_source(
          0.2, zero_background, source_fields, source_derivatives);
  CHECK_COMPLEX_NEAR(sharp_source.C12.c01, -device(correct), 0.0);
  CHECK(Kokkos::abs(sharp_source.C12.c01 -
                    same_mode_conjugate_source.C12.c01) > 0.1);
  teuk::ModePair malformed = pair;
  malformed.target = 4;
  bool rejected = false;
  try {
    (void)teuk::make_plus2_pair_lookup(registry, malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST_CASE("spin plus2 source is quadratic in common real amplitude") {
  std::mt19937_64 g(314159);
  const Fields f = random_fields(g);
  const Derivatives d = random_derivatives(g);
  const OuterDerivatives od = random_outer(g);
  const auto bg = teuk::kerr_background_point({1.0, 0.66, 1.3}, 0.24, 0.2,
                                               std::sqrt(0.96));
  constexpr double amplitude = 3.25;
  const auto base = teuk::plus2_compact_ordered_pair_source(0.24, bg, f, d);
  const auto changed = teuk::plus2_compact_ordered_pair_source(
      0.24, bg, scaled(f, amplitude), scaled(d, amplitude));
  const auto base_terms = flatten_pair(base, [](const KC& z) { return host(z); });
  const auto changed_terms =
      flatten_pair(changed, [](const KC& z) { return host(z); });
  for (const auto& [id, value] : base_terms) {
    CHECK(std::abs(changed_terms.at(id) - amplitude * amplitude * value) <
          3.0e-12);
  }
  const auto outer_base = teuk::plus2_compact_outer_source_over_r6(
      0.24, bg, base.J12.total(), base.K12.total(), base.Q12.total(), od);
  const auto outer_changed = teuk::plus2_compact_outer_source_over_r6(
      0.24, bg, changed.J12.total(), changed.K12.total(), changed.Q12.total(),
      scaled(od, amplitude * amplitude));
  const auto outer_base_terms =
      flatten_outer(outer_base, [](const KC& z) { return host(z); });
  const auto outer_changed_terms =
      flatten_outer(outer_changed, [](const KC& z) { return host(z); });
  for (const auto& [id, value] : outer_base_terms) {
    CHECK(std::abs(outer_changed_terms.at(id) -
                   amplitude * amplitude * value) < 5.0e-12);
  }
}

TEST_CASE("spin plus2 evolved forcing uses complete field normalization") {
  const KC source_over_r7(0.37, -0.28);
  const double radius = 0.19;
  const double cosine = -0.42;
  const double spin = 0.999;
  const double length = 1.6;
  const double D = std::pow(length, 4) +
                   spin * spin * radius * radius * cosine * cosine;
  const KC dminus(length * length, -spin * radius * cosine);
  const KC expected_factor = 2.0 * D * dminus * dminus * dminus * dminus;
  const KC actual = teuk::plus2_coordinate_forcing_from_source_over_r7(
      radius, cosine, spin, length, source_over_r7);
  const KC expected = expected_factor * source_over_r7;
  CHECK(Kokkos::abs(actual - expected) /
            std::max(Kokkos::abs(expected), 1.0) <
        5.0e-16);
  const KC scri_actual = teuk::plus2_coordinate_forcing_from_source_over_r7(
      0.0, cosine, spin, length, source_over_r7);
  const KC scri_expected =
      KC(2.0 * std::pow(length, 12), 0.0) * source_over_r7;
  CHECK_COMPLEX_NEAR(scri_actual, scri_expected, 2.0e-14);
}

TEST_CASE("spin plus2 S0 over R7 radial cancellation is exact and finite") {
  const KC J(0.37, -0.21), JT(-0.18, 0.44), JR(0.52, 0.09);
  const KC K(-0.31, 0.23), Q(0.11, -0.47), eth6K(0.63, 0.17);
  const int mode = -3;
  for (const auto parameters :
       {teuk::KerrParameters{1.0, 0.0, 1.3},
        teuk::KerrParameters{1.0, 0.67, 1.1},
        teuk::KerrParameters{1.0, 0.999, 1.0}}) {
    const double horizon =
        parameters.compactification_length *
        parameters.compactification_length /
        (parameters.mass +
         std::sqrt(parameters.mass * parameters.mass -
                   parameters.spin * parameters.spin));
    for (const double radius : {0.0, 0.23 * horizon, horizon}) {
      const double cosine = -0.41;
      const double sine = std::sqrt(1.0 - cosine * cosine);
      const auto background = teuk::kerr_background_point(
          parameters, radius, cosine, sine);
      const KC regularized =
          teuk::plus2_regularized_thorn5_j_minus_optical_over_r(
              J, JT, JR, mode, radius, cosine, parameters, background);
      const C expected_radial = host_regularized_radial_j(
          host(J), host(JT), host(JR), mode, radius, cosine, parameters,
          host(background));
      CHECK(std::abs(host(regularized) - expected_radial) < 3.0e-14);
      CHECK(std::isfinite(regularized.real()));
      CHECK(std::isfinite(regularized.imag()));
      const auto compact = teuk::plus2_compact_outer_source_over_r7(
          radius, background, K, Q,
          RegularizedOuterDerivatives{regularized, eth6K});
      const C expected = host_source_over_r7(
          radius, host(background), host(K), host(Q), expected_radial,
          host(eth6K));
      CHECK(std::abs(host(compact.total()) - expected) < 4.0e-14);
      if (radius > 0.0) {
        const KC thorn5 = teuk::thorn_n_point(
            J, JT, JR, 5, 2, 1, mode, radius, cosine, parameters.mass,
            parameters.spin, parameters.compactification_length,
            background.epsilon0);
        const auto raw = teuk::plus2_compact_outer_source_over_r6(
            radius, background, J, K, Q,
            OuterDerivatives{thorn5, eth6K});
        CHECK(std::abs(host(raw.total() / radius) - expected) < 8.0e-13);
      }
    }
  }
}

TEST_CASE("spin plus2 regularized S0 over R7 supports Jet and device parity") {
  using J1 = teuk::Jet1<KC>;
  const teuk::KerrParameters parameters{1.0, -0.74, 1.2};
  const double radius = 0.27, cosine = 0.36;
  const auto background = teuk::kerr_background_point(
      parameters, radius, cosine, std::sqrt(1.0 - cosine * cosine));
  const J1 value{KC(0.4, -0.2), KC(-0.13, 0.09)};
  const J1 dt{KC(-0.2, 0.5), KC(0.17, -0.11)};
  const J1 dr{KC(0.7, 0.1), KC(-0.08, 0.15)};
  const J1 host_result =
      teuk::plus2_regularized_thorn5_j_minus_optical_over_r(
          value, dt, dr, 2, radius, cosine, parameters, background);
  Kokkos::View<J1*> result("plus2_r7_device", 1);
  Kokkos::parallel_for(
      "plus2_r7_device_parity", 1, KOKKOS_LAMBDA(const int) {
        result(0) = teuk::plus2_regularized_thorn5_j_minus_optical_over_r(
            value, dt, dr, 2, radius, cosine, parameters, background);
      });
  const auto copy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         result);
  CHECK_COMPLEX_NEAR(copy(0).value, host_result.value, 2.0e-14);
  CHECK_COMPLEX_NEAR(copy(0).dt, host_result.dt, 2.0e-14);
  constexpr double epsilon = 2.0e-6;
  const auto shifted_result = [&](const double sign) {
    return teuk::plus2_regularized_thorn5_j_minus_optical_over_r(
        value.value + sign * epsilon * value.dt,
        dt.value + sign * epsilon * dt.dt,
        dr.value + sign * epsilon * dr.dt, 2, radius, cosine, parameters,
        background);
  };
  const KC finite_difference =
      (shifted_result(1.0) - shifted_result(-1.0)) / (2.0 * epsilon);
  CHECK_COMPLEX_NEAR(host_result.dt, finite_difference, 2.0e-10);
}

TEST_CASE("spin plus2 Jet tangent matches every directional derivative") {
  std::mt19937_64 g(271828);
  const Fields f = random_fields(g), df = random_fields(g);
  const Derivatives d = random_derivatives(g), dd = random_derivatives(g);
  const OuterDerivatives od = random_outer(g), dod = random_outer(g);
  const auto bg = teuk::kerr_background_point({1.0, -0.47, 1.4}, 0.29, -0.31,
                                               std::sqrt(1.0 - 0.31 * 0.31));
  const auto jsource = teuk::plus2_compact_ordered_pair_source(
      0.29, bg, make_jets(f, df), make_jets(d, dd));
  constexpr double epsilon = 2.0e-6;
  const auto plus = teuk::plus2_compact_ordered_pair_source(
      0.29, bg, shifted(f, df, epsilon), shifted(d, dd, epsilon));
  const auto minus = teuk::plus2_compact_ordered_pair_source(
      0.29, bg, shifted(f, df, -epsilon), shifted(d, dd, -epsilon));
  const auto analytic = flatten_pair(
      jsource, [](const teuk::Jet1<KC>& z) { return host(z.dt); });
  const auto plus_map = flatten_pair(plus, [](const KC& z) { return host(z); });
  const auto minus_map = flatten_pair(minus, [](const KC& z) { return host(z); });
  for (const auto& [id, tangent] : analytic) {
    const C finite_difference =
        (plus_map.at(id) - minus_map.at(id)) / (2.0 * epsilon);
    CHECK(std::abs(tangent - finite_difference) < 2.0e-9);
  }
  using J = teuk::Jet1<KC>;
  const teuk::Plus2OuterDerivativesT<J> jod{
      J(od.thorn5_J, dod.thorn5_J), J(od.eth6_K, dod.eth6_K)};
  const auto jouter = teuk::plus2_compact_outer_source_over_r6(
      0.29, bg, jsource.J12.total(), jsource.K12.total(),
      jsource.Q12.total(), jod);
  const auto outer_plus = teuk::plus2_compact_outer_source_over_r6(
      0.29, bg, plus.J12.total(), plus.K12.total(), plus.Q12.total(),
      shifted(od, dod, epsilon));
  const auto outer_minus = teuk::plus2_compact_outer_source_over_r6(
      0.29, bg, minus.J12.total(), minus.K12.total(), minus.Q12.total(),
      shifted(od, dod, -epsilon));
  const auto outer_analytic = flatten_outer(
      jouter, [](const J& z) { return host(z.dt); });
  const auto outer_plus_map =
      flatten_outer(outer_plus, [](const KC& z) { return host(z); });
  const auto outer_minus_map =
      flatten_outer(outer_minus, [](const KC& z) { return host(z); });
  for (const auto& [id, tangent] : outer_analytic) {
    const C finite_difference =
        (outer_plus_map.at(id) - outer_minus_map.at(id)) / (2.0 * epsilon);
    CHECK(std::abs(tangent - finite_difference) < 2.0e-9);
  }
}

TEST_CASE("spin plus2 standalone source has host device parity") {
  std::mt19937_64 g(8675309);
  const Fields f = random_fields(g);
  const Fields df = random_fields(g);
  const Derivatives d = random_derivatives(g);
  const Derivatives dd = random_derivatives(g);
  const OuterDerivatives od = random_outer(g);
  const auto bg = teuk::kerr_background_point({1.0, 0.91, 1.1}, 0.36, 0.42,
                                               std::sqrt(1.0 - 0.42 * 0.42));
  const Source host_pair =
      teuk::plus2_compact_ordered_pair_source(0.36, bg, f, d);
  const Outer host_outer = teuk::plus2_compact_outer_source_over_r6(
      0.36, bg, host_pair.J12.total(), host_pair.K12.total(),
      host_pair.Q12.total(), od);
  Kokkos::View<Source*> pair_result("plus2_pair_device_result", 1);
  Kokkos::View<Outer*> outer_result("plus2_outer_device_result", 1);
  using J = teuk::Jet1<KC>;
  using JetSource = teuk::Plus2OrderedPairSourceT<J>;
  const auto jf = make_jets(f, df);
  const auto jd = make_jets(d, dd);
  const JetSource host_jet =
      teuk::plus2_compact_ordered_pair_source(0.36, bg, jf, jd);
  Kokkos::View<JetSource*> jet_result("plus2_jet_device_result", 1);
  Kokkos::parallel_for(
      "plus2_standalone_device_parity", 1, KOKKOS_LAMBDA(const int) {
        const Source pair =
            teuk::plus2_compact_ordered_pair_source(0.36, bg, f, d);
        pair_result(0) = pair;
        outer_result(0) = teuk::plus2_compact_outer_source_over_r6(
            0.36, bg, pair.J12.total(), pair.K12.total(), pair.Q12.total(), od);
        jet_result(0) =
            teuk::plus2_compact_ordered_pair_source(0.36, bg, jf, jd);
      });
  const auto pair_copy = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pair_result);
  const auto outer_copy = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, outer_result);
  const auto jet_copy = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, jet_result);
  check_maps(flatten_pair(pair_copy(0), [](const KC& z) { return host(z); }),
             flatten_pair(host_pair, [](const KC& z) { return host(z); }),
             1.0e-14);
  check_maps(flatten_outer(outer_copy(0), [](const KC& z) { return host(z); }),
             flatten_outer(host_outer, [](const KC& z) { return host(z); }),
             1.0e-14);
  check_maps(flatten_pair(jet_copy(0), [](const J& z) { return host(z.value); }),
             flatten_pair(host_jet, [](const J& z) { return host(z.value); }),
             1.0e-14);
  check_maps(flatten_pair(jet_copy(0), [](const J& z) { return host(z.dt); }),
             flatten_pair(host_jet, [](const J& z) { return host(z.dt); }),
             1.0e-14);
}

TEST_CASE("spin plus2 point source performs no per-call allocation") {
  std::mt19937_64 g(424242);
  const Fields f = random_fields(g);
  const Derivatives d = random_derivatives(g);
  const OuterDerivatives od = random_outer(g);
  const auto bg = teuk::kerr_background_point({1.0, 0.52, 1.25}, 0.22, 0.1,
                                               std::sqrt(0.99));
  KC sink{};
  plus2_source_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_plus2_source_allocation);
  for (int repeat = 0; repeat < 64; ++repeat) {
    const auto pair =
        teuk::plus2_compact_ordered_pair_source(0.22, bg, f, d);
    const auto outer = teuk::plus2_compact_outer_source_over_r6(
        0.22, bg, pair.J12.total(), pair.K12.total(), pair.Q12.total(), od);
    sink += outer.total();
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(plus2_source_allocations == 0);
  CHECK(std::isfinite(sink.real()));
}

}  // namespace
