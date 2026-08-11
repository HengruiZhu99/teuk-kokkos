#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/angular.hpp"
#include "teuk/fields.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_io.hpp"
#include "teuk/types.hpp"

namespace teuk {

enum class PipelineField : std::size_t {
  FirstP = 0,
  FirstQ = 1,
  FirstPsi = 2,
  G = 3,
  Lambda = 4,
  H = 5,
  B = 6,
  Pi = 7,
  C = 8,
  U = 9,
  SecondP = 10,
  SecondQ = 11,
  SecondPsi = 12,
  Count = 13,
};

static_assert(static_cast<std::size_t>(PipelineField::Count) ==
              point_pipeline_field_count);

using PipelineModeView = Kokkos::View<int*, MemorySpace>;
using PipelineIndexView = Kokkos::View<std::size_t*, MemorySpace>;
using PipelineCoordinateView = Kokkos::View<Real*, MemorySpace>;
using PipelineFlatView = Kokkos::View<Complex*, MemorySpace>;
using UnmanagedPipelineFlatView =
    Kokkos::View<Complex*, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UnmanagedPipelineFieldView =
    Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// One allocation-owning coupled state in the documented
// (mode,field,radial,theta) ordering. All physics kernels consume unmanaged
// views of caller-supplied RK stages, so the same storage contract applies to
// state, k1--k4, and intermediate stages.
class SpatialPipelineStorage {
 public:
  SpatialPipelineStorage(const ModeRegistry& registry,
                         const UniformRadialGrid& radial_grid,
                         const angular::GaussLegendreGrid& angular_grid,
                         const std::string& label = "spatial_pipeline")
      : radial_grid_(radial_grid),
        mode_count_(registry.size()),
        theta_count_(angular_grid.size()),
        modes_(label + "_modes", registry.size()),
        sharp_indices_(label + "_sharp_indices", registry.size()),
        radius_(label + "_radius", radial_grid.size()),
        theta_(label + "_theta", angular_grid.size()),
        cos_theta_(label + "_cos_theta", angular_grid.size()),
        sin_theta_(label + "_sin_theta", angular_grid.size()),
        state_(label + "_state", registry.size(), point_pipeline_field_count,
               radial_grid.size(), angular_grid.size()),
        rhs_(label + "_rhs", registry.size(), point_pipeline_field_count,
             radial_grid.size(), angular_grid.size()) {
    if (!registry.is_closed_under_sharp()) {
      throw std::invalid_argument(
          "spatial pipeline modes must be closed under m -> -m");
    }
    if (angular_grid.size() == 0 ||
        angular_grid.weights.size() != angular_grid.size()) {
      throw std::invalid_argument("spatial pipeline angular grid is invalid");
    }
    auto host_modes = Kokkos::create_mirror_view(modes_);
    auto host_sharp = Kokkos::create_mirror_view(sharp_indices_);
    auto host_radius = Kokkos::create_mirror_view(radius_);
    auto host_theta = Kokkos::create_mirror_view(theta_);
    auto host_cos = Kokkos::create_mirror_view(cos_theta_);
    auto host_sin = Kokkos::create_mirror_view(sin_theta_);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      host_modes(mode) = registry.modes()[mode];
      host_sharp(mode) = registry.sharp_index(registry.modes()[mode]);
    }
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      host_radius(radial) = radial_grid.coordinate(radial);
    }
    for (std::size_t theta = 0; theta < angular_grid.size(); ++theta) {
      host_cos(theta) = angular_grid.x[theta];
      host_theta(theta) = angular_grid.theta(theta);
      host_sin(theta) =
          Kokkos::sqrt(1.0 - angular_grid.x[theta] * angular_grid.x[theta]);
    }
    Kokkos::deep_copy(modes_, host_modes);
    Kokkos::deep_copy(sharp_indices_, host_sharp);
    Kokkos::deep_copy(radius_, host_radius);
    Kokkos::deep_copy(theta_, host_theta);
    Kokkos::deep_copy(cos_theta_, host_cos);
    Kokkos::deep_copy(sin_theta_, host_sin);
  }

  [[nodiscard]] std::size_t mode_count() const { return mode_count_; }
  [[nodiscard]] std::size_t radial_count() const { return radial_grid_.size(); }
  [[nodiscard]] std::size_t theta_count() const { return theta_count_; }
  [[nodiscard]] std::size_t value_count() const { return state_.size(); }
  [[nodiscard]] const UniformRadialGrid& radial_grid() const {
    return radial_grid_;
  }
  [[nodiscard]] SnapshotShape snapshot_shape() const {
    return {mode_count_, point_pipeline_field_count, radial_count(),
            theta_count_};
  }

  [[nodiscard]] FieldView state() const { return state_; }
  [[nodiscard]] FieldView rhs() const { return rhs_; }
  [[nodiscard]] PipelineModeView modes() const { return modes_; }
  [[nodiscard]] PipelineIndexView sharp_indices() const {
    return sharp_indices_;
  }
  [[nodiscard]] PipelineCoordinateView radius() const { return radius_; }
  [[nodiscard]] PipelineCoordinateView theta() const { return theta_; }
  [[nodiscard]] PipelineCoordinateView cos_theta() const { return cos_theta_; }
  [[nodiscard]] PipelineCoordinateView sin_theta() const { return sin_theta_; }

  [[nodiscard]] UnmanagedPipelineFlatView flat_state() const {
    return UnmanagedPipelineFlatView(state_.data(), state_.size());
  }
  [[nodiscard]] UnmanagedPipelineFlatView flat_rhs() const {
    return UnmanagedPipelineFlatView(rhs_.data(), rhs_.size());
  }

  template <class FlatView>
  [[nodiscard]] UnmanagedPipelineFieldView reshape(
      const FlatView& flat) const {
    static_assert(FlatView::rank == 1, "pipeline RK stage must be flat");
    if (flat.extent(0) != value_count()) {
      throw std::invalid_argument("pipeline RK stage extent does not match");
    }
    return UnmanagedPipelineFieldView(
        flat.data(), mode_count_, point_pipeline_field_count, radial_count(),
        theta_count_);
  }

 private:
  UniformRadialGrid radial_grid_;
  std::size_t mode_count_;
  std::size_t theta_count_;
  PipelineModeView modes_;
  PipelineIndexView sharp_indices_;
  PipelineCoordinateView radius_;
  PipelineCoordinateView theta_;
  PipelineCoordinateView cos_theta_;
  PipelineCoordinateView sin_theta_;
  FieldView state_;
  FieldView rhs_;
};

}  // namespace teuk
