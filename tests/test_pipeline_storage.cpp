#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "teuk/angular.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_storage.hpp"

TEST_CASE("spatial pipeline storage preserves mode field radial theta order") {
  const teuk::ModeRegistry registry({2, -2, 0});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const auto angular_grid = teuk::angular::gauss_legendre(6);
  teuk::SpatialPipelineStorage storage(registry, radial_grid, angular_grid,
                                       "storage_order");
  CHECK(storage.snapshot_shape().modes == 3);
  CHECK(storage.snapshot_shape().fields == 13);
  CHECK(storage.snapshot_shape().radial == 9);
  CHECK(storage.snapshot_shape().theta == 6);
  CHECK(storage.value_count() == 3 * 13 * 9 * 6);

  const auto state = storage.state();
  Kokkos::parallel_for(
      "initialize_pipeline_order", storage.value_count(),
      KOKKOS_LAMBDA(const std::size_t flat) {
        state.data()[flat] = teuk::Complex(static_cast<double>(flat),
                                           -static_cast<double>(flat));
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        storage.state());
  for (std::size_t mode = 0; mode < storage.mode_count(); ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      for (std::size_t radial = 0; radial < storage.radial_count(); ++radial) {
        for (std::size_t theta = 0; theta < storage.theta_count(); ++theta) {
          const std::size_t flat =
              ((mode * teuk::point_pipeline_field_count + field) *
                   storage.radial_count() +
               radial) *
                  storage.theta_count() +
              theta;
          CHECK_COMPLEX_NEAR(
              host(mode, field, radial, theta),
              teuk::Complex(static_cast<double>(flat),
                            -static_cast<double>(flat)),
              0.0);
        }
      }
    }
  }
}

TEST_CASE("spatial pipeline flat RK views reshape without copy") {
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.7);
  teuk::SpatialPipelineStorage storage(
      registry, radial_grid, teuk::angular::gauss_legendre(5),
      "storage_reshape");
  const auto flat = storage.flat_state();
  const auto reshaped = storage.reshape(flat);
  CHECK(flat.data() == storage.state().data());
  CHECK(reshaped.data() == storage.state().data());
  Kokkos::deep_copy(flat, teuk::Complex(0.0, 0.0));
  Kokkos::parallel_for(
      "write_reshaped_pipeline", 1, KOKKOS_LAMBDA(const int) {
        reshaped(2, static_cast<std::size_t>(teuk::PipelineField::U), 4, 3) =
            teuk::Complex(7.0, -8.0);
      });
  const auto host_flat = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, flat);
  const std::size_t expected =
      ((2 * teuk::point_pipeline_field_count +
        static_cast<std::size_t>(teuk::PipelineField::U)) *
           storage.radial_count() +
       4) *
          storage.theta_count() +
      3;
  CHECK_COMPLEX_NEAR(host_flat(expected), teuk::Complex(7.0, -8.0), 0.0);
}

TEST_CASE("spatial pipeline rejects incomplete sharp mode storage") {
  bool rejected = false;
  try {
    const teuk::ModeRegistry registry({0, 1});
    const teuk::SpatialPipelineStorage storage(
        registry, teuk::UniformRadialGrid(9, 0.0, 0.7),
        teuk::angular::gauss_legendre(5));
    static_cast<void>(storage);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}
