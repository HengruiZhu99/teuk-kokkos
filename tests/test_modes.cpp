#include <complex>
#include <stdexcept>
#include <vector>

#include "test_harness.hpp"
#include "teuk/modes.hpp"

TEST_CASE("mode registry sorts signed modes and provides direct lookup") {
  const teuk::ModeRegistry registry({2, -1, 0, -2, 1});
  CHECK(registry.modes() == std::vector<int>({-2, -1, 0, 1, 2}));
  CHECK(registry.nonnegative_representatives() ==
        std::vector<int>({0, 1, 2}));
  CHECK(registry.index(-2) == 0);
  CHECK(registry.index(0) == 2);
  CHECK(registry.index(2) == 4);
  CHECK(registry.contains(-1));
  CHECK(!registry.contains(3));
  CHECK(registry.is_closed_under_sharp());
}

TEST_CASE("sharp lookup uses the negative mode and handles zero once") {
  const teuk::ModeRegistry registry({-2, 0, 2});
  const std::vector<std::complex<double>> values = {
      {1.0, 3.0}, {5.0, 7.0}, {11.0, 13.0}};

  const auto sharp_plus_two =
      std::conj(values.at(registry.sharp_index(+2)));
  const auto sharp_minus_two =
      std::conj(values.at(registry.sharp_index(-2)));
  const auto sharp_zero = std::conj(values.at(registry.sharp_index(0)));
  CHECK(sharp_plus_two == std::complex<double>(1.0, -3.0));
  CHECK(sharp_minus_two == std::complex<double>(11.0, -13.0));
  CHECK(sharp_zero == std::complex<double>(5.0, -7.0));
  CHECK(registry.sharp_index(0) == registry.index(0));
}

TEST_CASE("ordered mode pairs retain both factor orderings deterministically") {
  const teuk::ModeRegistry registry({-2, -1, 0, 1, 2}, {-2, 0, 2});
  const std::vector<teuk::ModePair> expected = {
      {-2, 0, -2, 0, 2, 0}, {-1, -1, -2, 1, 1, 0},
      {0, -2, -2, 2, 0, 0}, {-2, 2, 0, 0, 4, 2},
      {-1, 1, 0, 1, 3, 2},  {0, 0, 0, 2, 2, 2},
      {1, -1, 0, 3, 1, 2},  {2, -2, 0, 4, 0, 2},
      {0, 2, 2, 2, 4, 4},   {1, 1, 2, 3, 3, 4},
      {2, 0, 2, 4, 2, 4}};
  CHECK(registry.ordered_pairs() == expected);

  const auto zero_range = registry.pair_range(0);
  CHECK(zero_range.first == 3);
  CHECK(zero_range.second == 8);
  CHECK(registry.ordered_pairs()[zero_range.first].target == 0);
  CHECK(registry.ordered_pairs()[zero_range.second - 1].target == 0);
}

TEST_CASE("mode registry rejects ambiguous or incomplete configurations") {
  bool duplicate_rejected = false;
  try {
    const teuk::ModeRegistry registry({-1, 0, 0, 1});
    (void)registry;
  } catch (const std::invalid_argument&) {
    duplicate_rejected = true;
  }
  CHECK(duplicate_rejected);

  bool missing_target_rejected = false;
  try {
    const teuk::ModeRegistry registry({-1, 0, 1}, {0, 2});
    (void)registry;
  } catch (const std::invalid_argument&) {
    missing_target_rejected = true;
  }
  CHECK(missing_target_rejected);

  const teuk::ModeRegistry asymmetric({0, 1});
  CHECK(!asymmetric.is_closed_under_sharp());
  bool missing_sharp_rejected = false;
  try {
    (void)asymmetric.sharp_index(1);
  } catch (const std::out_of_range&) {
    missing_sharp_rejected = true;
  }
  CHECK(missing_sharp_rejected);
}

TEST_CASE("required mode validation preserves requested order") {
  const teuk::ModeRegistry registry({3, -3, 0, 1, -1});
  CHECK(registry.require_modes({1, -3, 0}) ==
        std::vector<std::size_t>({3, 0, 2}));
}

TEST_CASE("mode registry separates stored parents and quadratic targets") {
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-2, 2},
                                    {-4, 0, 4});
  CHECK(registry.parents() == std::vector<int>({-2, 2}));
  CHECK(registry.targets() == std::vector<int>({-4, 0, 4}));
  CHECK(registry.ordered_pairs().size() == 4);
  CHECK(registry.ordered_pairs()[0].m1 == -2);
  CHECK(registry.ordered_pairs()[0].m2 == -2);
  CHECK(registry.ordered_pairs()[0].target == -4);
  CHECK(registry.pair_range(0).second - registry.pair_range(0).first == 2);
  CHECK(registry.ordered_pairs().back().m1 == 2);
  CHECK(registry.ordered_pairs().back().m2 == 2);
  CHECK(registry.ordered_pairs().back().target == 4);
}

TEST_CASE("quadratic daughter completeness fails closed unless explicitly truncated") {
  const auto complete = teuk::validate_quadratic_daughter_modes(
      {-2, 2}, {-4, 0, 4}, false);
  CHECK(complete.complete());
  CHECK(complete.required == std::vector<int>({-4, 0, 4}));

  bool rejected = false;
  try {
    static_cast<void>(
        teuk::validate_quadratic_daughter_modes({-2, 2}, {0, 4}, false));
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("-4") != std::string::npos;
  }
  CHECK(rejected);
  const auto truncated =
      teuk::validate_quadratic_daughter_modes({-2, 2}, {0, 4}, true);
  CHECK(truncated.missing == std::vector<int>({-4}));
}
