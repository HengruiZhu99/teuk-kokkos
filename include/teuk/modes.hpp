#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace teuk {

// One contribution to a quadratic target mode. Indices refer directly to the
// compact storage owned by ModeRegistry; physical mode numbers are retained so
// diagnostics never have to reconstruct them from storage order.
struct ModePair {
  int m1;
  int m2;
  int target;
  std::size_t index1;
  std::size_t index2;
  std::size_t target_index;

  friend bool operator==(const ModePair&, const ModePair&) = default;
};

// Immutable-by-interface bookkeeping for explicitly stored signed m modes.
// Construction sorts the supplied lists, but rejects duplicates rather than
// silently changing a configuration. Pair ordering is target-major and then
// m1-major, and therefore independent of associative-container iteration.
class ModeRegistry {
 public:
  explicit ModeRegistry(std::vector<int> modes)
      : ModeRegistry(modes, modes) {}

  ModeRegistry(std::vector<int> modes, std::vector<int> targets)
      : modes_(sorted_unique(std::move(modes), "mode")),
        targets_(sorted_unique(std::move(targets), "target mode")) {
    if (modes_.empty()) {
      throw std::invalid_argument("ModeRegistry requires at least one mode");
    }
    if (targets_.empty()) {
      throw std::invalid_argument(
          "ModeRegistry requires at least one target mode");
    }

    min_mode_ = modes_.front();
    max_mode_ = modes_.back();
    const auto lookup_size =
        static_cast<std::size_t>(static_cast<long long>(max_mode_) -
                                 static_cast<long long>(min_mode_) + 1LL);
    index_by_offset_.assign(lookup_size, missing_index());
    for (std::size_t i = 0; i < modes_.size(); ++i) {
      index_by_offset_[offset(modes_[i])] = i;
      if (modes_[i] >= 0) representatives_.push_back(modes_[i]);
    }

    for (const int target : targets_) {
      if (!contains(target)) {
        throw std::invalid_argument("target mode " + std::to_string(target) +
                                    " is not stored");
      }
    }

    build_ordered_pairs();
  }

  [[nodiscard]] const std::vector<int>& modes() const noexcept {
    return modes_;
  }
  [[nodiscard]] const std::vector<int>& targets() const noexcept {
    return targets_;
  }
  [[nodiscard]] const std::vector<int>& nonnegative_representatives()
      const noexcept {
    return representatives_;
  }
  [[nodiscard]] const std::vector<ModePair>& ordered_pairs() const noexcept {
    return ordered_pairs_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return modes_.size(); }

  [[nodiscard]] bool contains(const int mode) const noexcept {
    if (mode < min_mode_ || mode > max_mode_) return false;
    return index_by_offset_[offset(mode)] != missing_index();
  }

  [[nodiscard]] std::size_t index(const int mode) const {
    if (!contains(mode)) {
      throw std::out_of_range("mode " + std::to_string(mode) +
                              " is not stored");
    }
    return index_by_offset_[offset(mode)];
  }

  // Storage index used by X_m^sharp = conjugate(X_{-m}). This deliberately
  // looks up -m; it is not the index of X_m followed by conjugation.
  [[nodiscard]] std::size_t sharp_index(const int mode) const {
    if (mode == std::numeric_limits<int>::min()) {
      throw std::out_of_range("cannot negate the minimum integer mode");
    }
    return index(-mode);
  }

  [[nodiscard]] bool is_closed_under_sharp() const noexcept {
    for (const int mode : modes_) {
      if (mode == std::numeric_limits<int>::min() || !contains(-mode)) {
        return false;
      }
    }
    return true;
  }

  // Validate required physical modes without making any assumption about their
  // storage positions. The returned indices preserve the caller's order.
  [[nodiscard]] std::vector<std::size_t> require_modes(
      const std::vector<int>& required) const {
    std::vector<std::size_t> indices;
    indices.reserve(required.size());
    for (const int mode : required) indices.push_back(index(mode));
    return indices;
  }

  // Half-open range in ordered_pairs() for one target. All target pairs are
  // contiguous, including an empty range when no stored inputs sum to target.
  [[nodiscard]] std::pair<std::size_t, std::size_t> pair_range(
      const int target) const {
    const auto target_position =
        std::lower_bound(targets_.begin(), targets_.end(), target);
    if (target_position == targets_.end() || *target_position != target) {
      throw std::out_of_range("mode " + std::to_string(target) +
                              " is not a target");
    }
    const auto i = static_cast<std::size_t>(target_position - targets_.begin());
    return {pair_offsets_[i], pair_offsets_[i + 1]};
  }

 private:
  static constexpr std::size_t missing_index() noexcept {
    return std::numeric_limits<std::size_t>::max();
  }

  static std::vector<int> sorted_unique(std::vector<int> values,
                                        const char* description) {
    std::sort(values.begin(), values.end());
    if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
      throw std::invalid_argument(std::string("duplicate ") + description);
    }
    return values;
  }

  [[nodiscard]] std::size_t offset(const int mode) const noexcept {
    return static_cast<std::size_t>(static_cast<long long>(mode) -
                                    static_cast<long long>(min_mode_));
  }

  void build_ordered_pairs() {
    pair_offsets_.reserve(targets_.size() + 1);
    for (const int target : targets_) {
      pair_offsets_.push_back(ordered_pairs_.size());
      for (const int m1 : modes_) {
        const long long m2_wide =
            static_cast<long long>(target) - static_cast<long long>(m1);
        if (m2_wide < std::numeric_limits<int>::min() ||
            m2_wide > std::numeric_limits<int>::max()) {
          continue;
        }
        const int m2 = static_cast<int>(m2_wide);
        if (!contains(m2)) continue;
        ordered_pairs_.push_back(
            {m1, m2, target, index(m1), index(m2), index(target)});
      }
    }
    pair_offsets_.push_back(ordered_pairs_.size());
  }

  std::vector<int> modes_;
  std::vector<int> targets_;
  std::vector<int> representatives_;
  std::vector<std::size_t> index_by_offset_;
  std::vector<ModePair> ordered_pairs_;
  std::vector<std::size_t> pair_offsets_;
  int min_mode_ = 0;
  int max_mode_ = 0;
};

}  // namespace teuk
