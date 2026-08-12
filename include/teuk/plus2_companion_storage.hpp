#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

#include "teuk/device_rk4.hpp"
#include "teuk/fields.hpp"
#include "teuk/linear_spatial.hpp"
#include "teuk/types.hpp"

namespace teuk {

using Plus2CompanionValueView =
    Kokkos::View<Complex***, Kokkos::LayoutRight, MemorySpace>;
using Plus2CompanionScratchView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace>;
using Plus2CompanionFlatView = Kokkos::View<
    Complex*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Allocation-owning architecture slice for a passive spin +2 companion.
//
// The disabled object contains a disengaged std::optional and therefore owns
// no Kokkos allocation.  It is intentionally not embedded in SpatialPipeline
// yet: doing so before the reviewed quadratic source and a common-stage
// coupled RK driver exist would invite operator splitting.  When enabled,
// this object provides the one (P_plus,Q_plus,Z_plus) state, the reusable
// radial/angular RHS scratch, and the four-stage RK storage needed by that
// future driver.  It contains no source formula and calls no evolution
// kernel; enabled View construction may perform backend initialization.
class Plus2CompanionStorage {
 public:
  Plus2CompanionStorage() = default;

  [[nodiscard]] static Plus2CompanionStorage enabled(
      const std::size_t mode_count, const std::size_t radial_count,
      const std::size_t theta_count,
      const std::string& label = "plus2_companion") {
    if (mode_count == 0 || radial_count == 0 || theta_count == 0) {
      throw std::invalid_argument(
          "enabled spin +2 companion extents must be nonzero");
    }
    Plus2CompanionStorage result;
    result.mode_count_ = mode_count;
    result.radial_count_ = radial_count;
    result.theta_count_ = theta_count;
    result.storage_.emplace(mode_count, radial_count, theta_count, label);
    return result;
  }

  [[nodiscard]] bool is_enabled() const noexcept {
    return storage_.has_value();
  }
  [[nodiscard]] std::size_t mode_count() const noexcept {
    return mode_count_;
  }
  [[nodiscard]] std::size_t radial_count() const noexcept {
    return radial_count_;
  }
  [[nodiscard]] std::size_t theta_count() const noexcept {
    return theta_count_;
  }
  [[nodiscard]] std::size_t value_count() const noexcept {
    return mode_count_ * static_cast<std::size_t>(TeukolskyField::Count) *
           radial_count_ * theta_count_;
  }

  [[nodiscard]] FieldView state() const { return get().state; }
  [[nodiscard]] Plus2CompanionValueView angular_laplacian() const {
    return get().angular_laplacian;
  }
  [[nodiscard]] Plus2CompanionScratchView radial_scratch() const {
    return get().radial_scratch;
  }
  [[nodiscard]] Plus2CompanionFlatView flat_state() const {
    const auto view = get().state;
    return Plus2CompanionFlatView(view.data(), view.size());
  }
  [[nodiscard]] DeviceRK4Workspace<Complex>& rk_workspace() {
    return get().rk_workspace;
  }
  [[nodiscard]] const DeviceRK4Workspace<Complex>& rk_workspace() const {
    return get().rk_workspace;
  }

 private:
  struct EnabledStorage {
    EnabledStorage(const std::size_t mode_count,
                   const std::size_t radial_count,
                   const std::size_t theta_count, const std::string& label)
        : state(label + "_state", mode_count,
                static_cast<std::size_t>(TeukolskyField::Count),
                radial_count, theta_count),
          angular_laplacian(label + "_angular_laplacian", mode_count,
                            radial_count, theta_count),
          radial_scratch(
              label + "_radial_scratch", mode_count,
              static_cast<std::size_t>(TeukolskyRadialScratch::Count),
              radial_count, theta_count),
          rk_workspace(mode_count *
                       static_cast<std::size_t>(TeukolskyField::Count) *
                       radial_count * theta_count) {}

    FieldView state;
    Plus2CompanionValueView angular_laplacian;
    Plus2CompanionScratchView radial_scratch;
    DeviceRK4Workspace<Complex> rk_workspace;
  };

  [[nodiscard]] EnabledStorage& get() {
    if (!storage_) {
      throw std::logic_error("spin +2 companion storage is disabled");
    }
    return *storage_;
  }
  [[nodiscard]] const EnabledStorage& get() const {
    if (!storage_) {
      throw std::logic_error("spin +2 companion storage is disabled");
    }
    return *storage_;
  }

  std::size_t mode_count_ = 0;
  std::size_t radial_count_ = 0;
  std::size_t theta_count_ = 0;
  std::optional<EnabledStorage> storage_;
};

}  // namespace teuk
