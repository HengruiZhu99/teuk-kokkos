#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_bianchi_transport.hpp"
#include "teuk/plus2_bianchi_transport_checkpoint.hpp"

namespace {

using C = teuk::Complex;
using execution_space = teuk::ExecutionSpace;
using transport_type = teuk::Plus2BianchiTransport<execution_space>;

constexpr int ell_max = 3;
constexpr int theta_count = 6;
constexpr std::size_t radial_count = 24;
constexpr double primary_rate = 0.17;

int bianchi_transport_allocations = 0;
int bianchi_transport_fences = 0;

void count_bianchi_transport_allocation(Kokkos::Tools::SpaceHandle,
                                        const char*, const void*,
                                        std::uint64_t) {
  ++bianchi_transport_allocations;
}

void count_bianchi_transport_fence(const char*, std::uint32_t,
                                   std::uint64_t*) {
  ++bianchi_transport_fences;
}

struct TransportFixture {
  execution_space execution;
  teuk::ModeRegistry registry{{0}};
  teuk::UniformRadialGrid grid;
  teuk::KerrParameters parameters;
  teuk::Plus2BianchiThetaView cos_theta{"transport_cos", theta_count};
  teuk::Plus2BianchiThetaView sin_theta{"transport_sin", theta_count};
  std::unique_ptr<transport_type> transport;

  explicit TransportFixture(const double amplitude = 1.0,
                            const bool initialize = true,
                            const std::size_t points = radial_count,
                            const teuk::KerrParameters parameters_in =
                                {1.0, 0.91, 1.3})
      : grid(points, 0.0, 0.68), parameters(parameters_in) {
    const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
    auto host_cos = Kokkos::create_mirror_view(cos_theta);
    auto host_sin = Kokkos::create_mirror_view(sin_theta);
    for (int theta = 0; theta < theta_count; ++theta) {
      host_cos(static_cast<std::size_t>(theta)) = angular_grid.x[theta];
      host_sin(static_cast<std::size_t>(theta)) =
          std::sqrt(std::max(0.0, 1.0 - angular_grid.x[theta] *
                                           angular_grid.x[theta]));
    }
    Kokkos::deep_copy(execution, cos_theta, host_cos);
    Kokkos::deep_copy(execution, sin_theta, host_sin);
    execution.fence("initialize Bianchi transport coordinates");
    transport = std::make_unique<transport_type>(
        execution, registry, grid, parameters, ell_max, cos_theta, sin_theta,
        teuk::RadialDiscretization::D105, "test_bianchi_transport");
    if (initialize) initialize_state(amplitude);
  }

  void initialize_state(const double amplitude) {
    teuk::Plus2BianchiStateView initial(
        "transport_initial", registry.size(),
        static_cast<std::size_t>(teuk::Plus2BianchiStateComponent::Count),
        grid.size(), theta_count);
    const auto cos = cos_theta;
    const auto radial_grid = grid;
    Kokkos::parallel_for(
        "initialize_manufactured_bianchi_state",
        Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>(
            execution, {0, 0},
            {static_cast<long>(grid.size()), static_cast<long>(theta_count)}),
        KOKKOS_LAMBDA(const std::size_t radial, const std::size_t theta) {
          const double radius = radial_grid.coordinate(radial);
          const double x = cos(theta);
          initial(0, static_cast<std::size_t>(
                         teuk::Plus2BianchiStateComponent::Z0),
                  radial, theta) =
              amplitude * C((0.12 + 0.03 * radius) * (1.0 - x * x),
                            -0.025 * (1.0 + radius) * x);
          initial(0, static_cast<std::size_t>(
                         teuk::Plus2BianchiStateComponent::Z1),
                  radial, theta) =
              amplitude * C((0.08 - 0.02 * radius) * x,
                            0.04 * (1.0 + 0.1 * radius) * (1.0 - x * x));
        });
    transport->initialize(
        execution, initial,
        {"manufactured-curvature-and-boundary-v1", true, true, true,
         "manufactured-exact-boundary-v1"});
    execution.fence("finish Bianchi transport initialization");
  }

  [[nodiscard]] teuk::Plus2BianchiStageCapability capability() const {
    return {"manufactured-exact-boundary-v1", true, true, true, true, true,
            teuk::RadialDiscretization::D105};
  }
};

struct WriteManufacturedPrimaryFunctor {
  C* values;
  std::uint64_t* stamps;
  const C* primary;
  double amplitude;
  std::uint64_t generation;
  std::size_t radial;
  std::size_t theta_count;
  bool omit_stamp;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t fields = static_cast<std::size_t>(
        teuk::Plus2BianchiPrimaryComponent::Count);
    const std::size_t plane = radial * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial_index = within / theta_count;
    const std::size_t theta = within - radial_index * theta_count;
    const double x = primary[0].real();
    const double shape = 1.0 + 0.013 * static_cast<double>(radial_index) +
                         0.021 * static_cast<double>(theta);
    const C h = amplitude * x * C(0.035 * shape, -0.012 * shape);
    const C c = amplitude * x * C(-0.017 * shape, 0.009 * shape);
    const C b = amplitude * x * C(0.011 * shape, 0.014 * shape);
    const C ta = amplitude * x * C(-0.006 * shape, 0.003 * shape);
    const C sig = amplitude * x * C(0.019 * shape, -0.008 * shape);
    const C input[fields]{h,
                          primary_rate * h,
                          primary_rate * primary_rate * h,
                          c,
                          primary_rate * c,
                          b,
                          primary_rate * b,
                          ta,
                          primary_rate * ta,
                          sig,
                          primary_rate * sig,
                          C{},
                          C{}};
    for (std::size_t field = 0; field < fields; ++field) {
      const std::size_t index =
          ((mode * fields + field) * radial + radial_index) * theta_count +
          theta;
      values[index] = input[field];
      if (!(omit_stamp && mode == 0 && radial_index == 0 && theta == 0 &&
            field + 1 == fields)) {
        stamps[index] = generation;
      }
    }
  }
};

struct ManufacturedPrimaryRhsFunctor {
  const C* input;
  C* output;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    output[i] = primary_rate * input[i];
  }
};

struct WriteSigmaOnlyPrimaryFunctor {
  C* values;
  std::uint64_t* stamps;
  C sigma;
  C sigma_t;
  std::uint64_t generation;
  std::size_t point_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t fields = static_cast<std::size_t>(
        teuk::Plus2BianchiPrimaryComponent::Count);
    for (std::size_t field = 0; field < fields; ++field) {
      const std::size_t index = field * point_count + flat;
      C value{};
      if (field == static_cast<std::size_t>(
                       teuk::Plus2BianchiPrimaryComponent::Sig)) {
        value = sigma;
      } else if (field == static_cast<std::size_t>(
                              teuk::Plus2BianchiPrimaryComponent::SigT)) {
        value = sigma_t;
      }
      values[index] = value;
      stamps[index] = generation;
    }
  }
};

static_assert(std::is_trivially_copyable_v<WriteManufacturedPrimaryFunctor>);
static_assert(std::is_trivially_copyable_v<ManufacturedPrimaryRhsFunctor>);
static_assert(std::is_trivially_copyable_v<WriteSigmaOnlyPrimaryFunctor>);

auto complete_primary_producer(const double amplitude = 1.0,
                               const bool omit_stamp = false) {
  return [=](const execution_space& execution, const double,
             const auto& primary_stage,
             const teuk::Plus2BianchiPrimaryWriteTarget target) {
    using primary_view_type =
        std::remove_cvref_t<decltype(primary_stage)>;
    static_assert(std::is_const_v<typename primary_view_type::value_type>);
    const std::size_t modes = target.fields.extent(0);
    const std::size_t radial = target.fields.extent(2);
    const std::size_t theta_count_value = target.fields.extent(3);
    Kokkos::parallel_for(
        "write_manufactured_bianchi_primary",
        Kokkos::RangePolicy<execution_space>(
            execution, 0, modes * radial * theta_count_value),
        WriteManufacturedPrimaryFunctor{
            target.fields.data(), target.stamps.data(), primary_stage.data(),
            amplitude, target.generation, radial, theta_count_value,
            omit_stamp});
  };
}

auto primary_rhs() {
  return [](const execution_space& execution, const double, const auto& input,
            const auto& output) {
    Kokkos::parallel_for(
        "manufactured_bianchi_primary_rhs",
        Kokkos::RangePolicy<execution_space>(execution, 0, input.extent(0)),
        ManufacturedPrimaryRhsFunctor{input.data(), output.data()});
  };
}

std::vector<C> copy_state(const transport_type& transport) {
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, transport.state());
  return std::vector<C>(host.data(), host.data() + host.size());
}

double difference_norm(const std::vector<C>& left,
                       const std::vector<C>& right) {
  double sum = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    sum += Kokkos::abs(left[i] - right[i]) *
           Kokkos::abs(left[i] - right[i]);
  }
  return std::sqrt(sum / static_cast<double>(left.size()));
}

struct InitializeSpatialBianchiStateFunctor {
  C* state;
  teuk::UniformRadialGrid grid;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t radial = flat / theta_count;
    const double radius = grid.coordinate(radial);
    state[flat] = C(0.12, -0.07) * Kokkos::exp(3.0 * radius);
    state[radial_count * theta_count + flat] = C{};
  }
};

struct WriteSpatialBianchiPrimaryFunctor {
  C* values;
  std::uint64_t* stamps;
  teuk::UniformRadialGrid grid;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t fields = static_cast<std::size_t>(
        teuk::Plus2BianchiPrimaryComponent::Count);
    const std::size_t radial = flat / theta_count;
    const double radius = grid.coordinate(radial);
    const C sigma = C(0.03, 0.01) *
                    (1.0 + 0.2 * radius + 0.05 * radius * radius);
    const C sigma_t = C(-0.02, 0.015) * (1.0 - 0.1 * radius);
    for (std::size_t field = 0; field < fields; ++field) {
      const std::size_t index = field * radial_count * theta_count + flat;
      values[index] =
          field == static_cast<std::size_t>(
                       teuk::Plus2BianchiPrimaryComponent::Sig)
              ? sigma
              : field == static_cast<std::size_t>(
                             teuk::Plus2BianchiPrimaryComponent::SigT)
                    ? sigma_t
                    : C{};
      stamps[index] = generation;
    }
  }
};

double spatial_bianchi_z0tt_error(const std::size_t points) {
  const teuk::KerrParameters parameters{1.0, 0.0, 1.3};
  TransportFixture fixture(1.0, false, points, parameters);
  teuk::Plus2BianchiStateView state(
      "spatial Bianchi state", fixture.registry.size(),
      static_cast<std::size_t>(teuk::Plus2BianchiStateComponent::Count), points,
      theta_count);
  Kokkos::parallel_for(
      "initialize_spatial_Bianchi_state",
      Kokkos::RangePolicy<execution_space>(fixture.execution, 0,
                                           points * theta_count),
      InitializeSpatialBianchiStateFunctor{state.data(), fixture.grid, points,
                                           theta_count});
  fixture.transport->initialize(
      fixture.execution, state,
      {"analytic-spatial-bianchi-v1", true, true, true,
       "analytic-spatial-boundary-v1"});
  auto capability = fixture.capability();
  capability.boundary_evidence_id = "analytic-spatial-boundary-v1";
  Kokkos::View<C*> primary("spatial Bianchi primary", 1);
  Kokkos::View<C*> rhs("spatial Bianchi rhs", fixture.transport->state().size());
  const auto state_view = fixture.transport->state();
  const transport_type::const_flat_view flat_state(state_view.data(),
                                                    state_view.size());
  const transport_type::flat_view flat_rhs(rhs.data(), rhs.size());
  const auto radial_grid = fixture.grid;
  const auto producer = [=](const execution_space& execution, const double,
                            const auto&,
                            const teuk::Plus2BianchiPrimaryWriteTarget target) {
    Kokkos::parallel_for(
        "write_spatial_Bianchi_primary",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             points * theta_count),
        WriteSpatialBianchiPrimaryFunctor{
            target.fields.data(), target.stamps.data(), radial_grid,
            target.generation, points, theta_count});
  };
  fixture.transport->evaluate_stage(
      fixture.execution, 0.0, primary, flat_state, flat_rhs,
      fixture.transport->next_generation(), capability, producer);
  fixture.execution.fence("finish spatial Bianchi stage");
  const auto curvature = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.transport->latest_stage().fields);

  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double psi20 = -parameters.mass /
                       (length2 * length2 * length2);
  double error = 0.0;
  for (std::size_t radial = 0; radial < points; ++radial) {
    const double radius = fixture.grid.coordinate(radial);
    constexpr double wave_number = 3.0;
    const C z0 = C(0.12, -0.07) * std::exp(wave_number * radius);
    const C sigma = C(0.03, 0.01) *
                    (1.0 + 0.2 * radius + 0.05 * radius * radius);
    const C sigma_r = C(0.03, 0.01) * (0.2 + 0.1 * radius);
    const C sigma_t = C(-0.02, 0.015) * (1.0 - 0.1 * radius);
    const double time_coefficient =
        2.0 + 4.0 * parameters.mass * radius / length2;
    const double time_coefficient_r = 4.0 * parameters.mass / length2;
    const C q = wave_number * radius * z0 + 4.0 * z0;
    const C q_r = wave_number * wave_number * radius * z0 +
                  5.0 * wave_number * z0;
    const C numerator = -radius * q / length2 + 3.0 * sigma * psi20;
    const C numerator_r = -(q + radius * q_r) / length2 +
                          3.0 * sigma_r * psi20;
    const C z0_t = numerator / time_coefficient;
    const C z0_t_r =
        (numerator_r * time_coefficient -
         numerator * time_coefficient_r) /
        (time_coefficient * time_coefficient);
    const C f0_t = radius * z0_t / length2 + 3.0 * sigma_t * psi20;
    const C expected =
        (f0_t - radius / length2 * (radius * z0_t_r + 5.0 * z0_t)) /
        time_coefficient;
    for (std::size_t theta = 0; theta < theta_count; ++theta) {
      error = std::max(
          error,
          Kokkos::abs(
              curvature(
                  0,
                  static_cast<std::size_t>(
                      teuk::Plus2TransportedCurvatureComponent::Z0TT),
                  radial, theta) -
              expected));
    }
  }
  return error;
}

struct TemporaryCheckpoint {
  TemporaryCheckpoint()
      : path(std::filesystem::temp_directory_path() /
             ("teuk-bianchi-transport-" +
              std::to_string(std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count()) +
              ".bin")) {}
  ~TemporaryCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(path.string() + ".tmp", ignored);
  }
  std::filesystem::path path;
};

std::vector<C> evolve_manufactured(const int steps,
                                   std::vector<std::uint64_t>* generations =
                                       nullptr,
                                   std::vector<double>* stage_times = nullptr,
                                   const double initial_amplitude = 1.0,
                                   const double producer_amplitude = 1.0) {
  TransportFixture fixture(initial_amplitude);
  Kokkos::View<C*> primary("Bianchi manufactured primary", 1);
  Kokkos::deep_copy(primary, C(initial_amplitude, 0.0));
  teuk::DeviceRK4Workspace<C, execution_space> primary_workspace(1);
  auto producer = complete_primary_producer(producer_amplitude);
  auto observer = [&](const execution_space&, const double stage_time,
                      const std::uint64_t generation,
                      const teuk::Plus2TransportedCurvatureStage&,
                      const teuk::Plus2BianchiDerivativeStage&) {
    if (generations) generations->push_back(generation);
    if (stage_times) stage_times->push_back(stage_time);
  };
  const double final_time = 0.2;
  const double step = final_time / static_cast<double>(steps);
  for (int n = 0; n < steps; ++n) {
    teuk::device_one_way_bianchi_transport_rk4_step(
        fixture.execution, primary, static_cast<double>(n) * step, step,
        primary_rhs(), producer, observer, primary_workspace,
        *fixture.transport, fixture.capability());
  }
  fixture.execution.fence("finish manufactured Bianchi RK4");
  return copy_state(*fixture.transport);
}

}  // namespace

TEST_CASE("plus2 Bianchi transport has fourth-order common-stage time RK4") {
  std::vector<std::uint64_t> generations;
  std::vector<double> stage_times;
  const auto coarse = evolve_manufactured(4, &generations, &stage_times);
  const auto medium = evolve_manufactured(8);
  const auto fine = evolve_manufactured(16);
  const auto reference = evolve_manufactured(64);
  const double coarse_error = difference_norm(coarse, reference);
  const double medium_error = difference_norm(medium, reference);
  const double fine_error = difference_norm(fine, reference);
  CHECK(coarse_error / medium_error > 13.0);
  CHECK(medium_error / fine_error > 13.0);
  CHECK(generations.size() == 16);
  for (std::size_t stage = 0; stage < generations.size(); ++stage) {
    CHECK(generations[stage] == stage + 1);
  }
  const std::array<double, 4> first_times{0.0, 0.025, 0.025, 0.05};
  for (std::size_t stage = 0; stage < first_times.size(); ++stage) {
    CHECK_NEAR(stage_times[stage], first_times[stage], 2.0e-16);
  }
}

TEST_CASE("plus2 Bianchi D10-5 nested radial transport is fourth order") {
  const double coarse = spatial_bianchi_z0tt_error(33);
  const double medium = spatial_bianchi_z0tt_error(65);
  const double fine = spatial_bianchi_z0tt_error(129);
  std::cout << "Bianchi D105 nested Z0TT endpoint-inclusive errors " << coarse
            << " " << medium << " " << fine << " ratios "
            << coarse / medium << " " << medium / fine << '\n';
  CHECK(coarse / medium >= 15.0);
  CHECK(medium / fine >= 15.0);
}

TEST_CASE("plus2 Bianchi adapter stamps every common RK stage on device") {
  TransportFixture fixture;
  Kokkos::View<C*> primary("Bianchi stamped primary", 1);
  Kokkos::deep_copy(primary, C(1.0, 0.0));
  teuk::DeviceRK4Workspace<C, execution_space> workspace(1);
  constexpr std::size_t component_count = static_cast<std::size_t>(
      teuk::Plus2TransportedCurvatureComponent::Count);
  Kokkos::View<std::uint64_t*> observed("Bianchi observed stage stamps",
                                        4 * component_count);
  std::size_t stage_slot = 0;
  auto observer = [&](const execution_space& execution, const double,
                      const std::uint64_t,
                      const teuk::Plus2TransportedCurvatureStage& stage,
                      const teuk::Plus2BianchiDerivativeStage& derivatives) {
    const std::size_t slot = stage_slot++;
    const auto stamps = stage.stamps;
    Kokkos::parallel_for(
        "audit_Bianchi_common_stage_stamp",
        Kokkos::RangePolicy<execution_space>(execution, 0, component_count),
        KOKKOS_LAMBDA(const std::size_t field) {
          observed(slot * component_count + field) = stamps(0, field, 0, 0);
        });
    const auto derivative_stamps = derivatives.stamps;
    Kokkos::parallel_for(
        "audit_Bianchi_derivative_common_stage_stamp",
        Kokkos::RangePolicy<execution_space>(execution, 0,
                                             derivative_stamps.extent(1)),
        KOKKOS_LAMBDA(const std::size_t field) {
          if (derivative_stamps(0, field, 0, 0) != slot + 1) {
            observed(slot * component_count) = 0;
          }
        });
  };
  auto producer = complete_primary_producer();
  teuk::device_one_way_bianchi_transport_rk4_step(
      fixture.execution, primary, 0.0, 0.01, primary_rhs(), producer,
      observer, workspace, *fixture.transport, fixture.capability());
  fixture.execution.fence("finish common-stage stamp audit");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        observed);
  CHECK(stage_slot == 4);
  for (std::size_t stage = 0; stage < 4; ++stage) {
    for (std::size_t field = 0; field < component_count; ++field) {
      CHECK(host(stage * component_count + field) == stage + 1);
    }
  }
}

TEST_CASE("plus2 Bianchi transport self-residuals close and scale linearly") {
  TransportFixture base;
  Kokkos::View<C*> primary("Bianchi amplitude primary", 1);
  Kokkos::deep_copy(primary, C(1.0, 0.0));
  teuk::DeviceRK4Workspace<C, execution_space> workspace(1);
  auto producer = complete_primary_producer();
  auto observer = [](const auto&, double, std::uint64_t, const auto&,
                     const auto&) {};
  teuk::device_one_way_bianchi_transport_rk4_step(
      base.execution, primary, 0.0, 0.01, primary_rhs(), producer, observer,
      workspace, *base.transport, base.capability());
  base.execution.fence("finish Bianchi constraint test");
  const auto constraints = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, base.transport->constraints());
  double maximum_constraint = 0.0;
  for (std::size_t i = 0; i < constraints.size(); ++i) {
    maximum_constraint = std::max(maximum_constraint,
                                  Kokkos::abs(constraints.data()[i]));
  }
  CHECK(maximum_constraint < 2.0e-13);
  const auto derivative_stage = base.transport->latest_derivative_stage();
  const auto derivative_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, derivative_stage.stamps);
  for (std::size_t i = 0; i < derivative_stamps.size(); ++i) {
    CHECK(derivative_stamps.data()[i] == 4);
  }

  const auto unscaled = evolve_manufactured(1);
  constexpr double scale = -1.7;
  const auto scaled = evolve_manufactured(1, nullptr, nullptr, scale, 1.0);
  double maximum = 0.0;
  for (std::size_t i = 0; i < unscaled.size(); ++i) {
    maximum = std::max(maximum, Kokkos::abs(unscaled[i]));
    CHECK_COMPLEX_NEAR(scaled[i], scale * unscaled[i], 2.0e-12);
  }
  CHECK(maximum > 1.0e-5);
}

TEST_CASE("plus2 Bianchi transport couples Sigma only to background psi20") {
  TransportFixture fixture(1.0, false);
  teuk::Plus2BianchiStateView zero_state(
      "zero Bianchi curvature state", fixture.registry.size(),
      static_cast<std::size_t>(teuk::Plus2BianchiStateComponent::Count),
      fixture.grid.size(), theta_count);
  Kokkos::deep_copy(fixture.execution, zero_state, C{});
  fixture.transport->initialize(
      fixture.execution, zero_state,
      {"sigma-background-curvature-regression-v1", true, true, true,
       "manufactured-exact-boundary-v1"});

  constexpr C sigma(0.071, -0.029);
  constexpr C sigma_t(-0.037, 0.016);
  Kokkos::View<C*> primary("sigma-only Bianchi primary", 1);
  Kokkos::View<C*> rhs("sigma-only Bianchi rhs",
                       fixture.transport->state().size());
  Kokkos::deep_copy(fixture.execution, primary, C{});
  const auto state_view = fixture.transport->state();
  const transport_type::const_flat_view state(state_view.data(),
                                               state_view.size());
  const transport_type::flat_view output(rhs.data(), rhs.size());
  const auto producer = [=](const execution_space& execution, const double,
                            const auto&,
                            const teuk::Plus2BianchiPrimaryWriteTarget target) {
    const std::size_t points = target.fields.extent(0) *
                               target.fields.extent(2) *
                               target.fields.extent(3);
    Kokkos::parallel_for(
        "write_sigma_only_bianchi_primary",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        WriteSigmaOnlyPrimaryFunctor{target.fields.data(),
                                     target.stamps.data(), sigma, sigma_t,
                                     target.generation, points});
  };
  fixture.transport->evaluate_stage(
      fixture.execution, 0.0, primary, state, output,
      fixture.transport->next_generation(), fixture.capability(), producer);
  fixture.execution.fence("finish Sigma psi20 Bianchi regression");

  const auto derivatives = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.transport->latest_derivative_stage().fields);
  const std::size_t radial = radial_count / 2;
  const std::size_t theta = theta_count / 2;
  const double radius = fixture.grid.coordinate(radial);
  const auto background = teuk::kerr_background_point(
      fixture.parameters, radius,
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                          fixture.cos_theta)(theta),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                          fixture.sin_theta)(theta));
  const C expected_f0 = 3.0 * sigma * background.psi20;
  const double length2 = fixture.parameters.compactification_length *
                         fixture.parameters.compactification_length;
  const C expected_z0_t =
      expected_f0 /
      (2.0 + 4.0 * fixture.parameters.mass * radius / length2);
  const C expected_f0_t =
      -radius * background.mu0 * expected_z0_t +
      3.0 * sigma_t * background.psi20;
  CHECK_COMPLEX_NEAR(
      derivatives(0, static_cast<std::size_t>(
                         teuk::Plus2BianchiDerivativeComponent::CapitalDelta5Z0),
                  radial, theta),
      expected_f0, 3.0e-14);
  CHECK_COMPLEX_NEAR(
      derivatives(
          0,
          static_cast<std::size_t>(
              teuk::Plus2BianchiDerivativeComponent::CapitalDelta5Z0T),
          radial, theta),
      expected_f0_t, 3.0e-14);
  CHECK(Kokkos::abs(expected_f0) > 1.0e-5);
  CHECK(Kokkos::abs(expected_f0_t) > 1.0e-5);
}

TEST_CASE("plus2 Bianchi checkpoint round trip validates before mutation") {
  TransportFixture source;
  Kokkos::View<C*> primary("checkpoint Bianchi primary", 1);
  Kokkos::deep_copy(primary, C(1.0, 0.0));
  teuk::DeviceRK4Workspace<C, execution_space> workspace(1);
  auto producer = complete_primary_producer();
  auto observer = [](const auto&, double, std::uint64_t, const auto&,
                     const auto&) {};
  teuk::device_one_way_bianchi_transport_rk4_step(
      source.execution, primary, 0.0, 0.01, primary_rhs(), producer, observer,
      workspace, *source.transport, source.capability());
  source.execution.fence("finish checkpoint source step");
  const auto expected_state = copy_state(*source.transport);

  const teuk::Plus2BianchiCheckpointProvenance provenance{
      "b48b20c", 1, "manufactured-exact-boundary-v1"};
  TemporaryCheckpoint mismatched_boundary_checkpoint;
  bool save_boundary_rejected = false;
  try {
    static_cast<void>(teuk::save_plus2_bianchi_transport_checkpoint(
        source.execution, mismatched_boundary_checkpoint.path,
        *source.transport, {0.01, 1},
        {"b48b20c", 1, "different-boundary-evidence"}));
  } catch (const std::invalid_argument&) {
    save_boundary_rejected = true;
  }
  CHECK(save_boundary_rejected);
  TemporaryCheckpoint checkpoint;
  const auto saved = teuk::save_plus2_bianchi_transport_checkpoint(
      source.execution, checkpoint.path, *source.transport, {0.01, 1},
      provenance);
  CHECK(saved.last_generation == 4);
  CHECK(saved.state_checksum != 0);
  for (const double bad_spin : {
           std::numeric_limits<double>::infinity(),
           saved.background.mass + 0.1}) {
    auto invalid_metadata = saved;
    invalid_metadata.background.spin = bad_spin;
    bool invalid_rejected = false;
    try {
      teuk::plus2_bianchi_checkpoint_detail::validate_metadata(
          invalid_metadata);
    } catch (const std::invalid_argument&) {
      invalid_rejected = true;
    }
    CHECK(invalid_rejected);
  }

  TransportFixture restored(1.0, false);
  const auto expectations = teuk::plus2_bianchi_checkpoint_expectations(
      *restored.transport, provenance,
      "manufactured-curvature-and-boundary-v1", true);
  const auto loaded = teuk::load_plus2_bianchi_transport_checkpoint(
      restored.execution, checkpoint.path, *restored.transport, expectations);
  CHECK(loaded.progress.step == 1);
  CHECK_NEAR(loaded.progress.time, 0.01, 0.0);
  CHECK(restored.transport->last_generation() == 4);
  bool transient_stage_rejected = false;
  try {
    static_cast<void>(restored.transport->latest_stage());
  } catch (const std::logic_error&) {
    transient_stage_rejected = true;
  }
  CHECK(transient_stage_rejected);
  const auto actual_state = copy_state(*restored.transport);
  CHECK(actual_state == expected_state);

  TransportFixture sentinel(-2.0);
  const auto before = copy_state(*sentinel.transport);
  auto mismatch = teuk::plus2_bianchi_checkpoint_expectations(
      *sentinel.transport, provenance,
      "manufactured-curvature-and-boundary-v1", true);
  mismatch.git_commit = "deadbee";
  bool rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_bianchi_transport_checkpoint(
        sentinel.execution, checkpoint.path, *sentinel.transport, mismatch));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(copy_state(*sentinel.transport) == before);

  mismatch = teuk::plus2_bianchi_checkpoint_expectations(
      *sentinel.transport, provenance,
      "manufactured-curvature-and-boundary-v1", true);
  mismatch.boundary_evidence_id = "different-boundary-evidence";
  rejected = false;
  try {
    static_cast<void>(teuk::load_plus2_bianchi_transport_checkpoint(
        sentinel.execution, checkpoint.path, *sentinel.transport, mismatch));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(copy_state(*sentinel.transport) == before);
}

TEST_CASE("plus2 Bianchi transport initialization and stages fail closed") {
  TransportFixture fixture(1.0, false);
  teuk::Plus2BianchiStateView initial(
      "rejected Bianchi initial", 1,
      static_cast<std::size_t>(teuk::Plus2BianchiStateComponent::Count),
      radial_count, theta_count);
  bool rejected = false;
  try {
    fixture.transport->initialize(
        fixture.execution, initial,
        {"missing-scri-evidence", true, true, false,
         "manufactured-exact-boundary-v1"});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  fixture.initialize_state(1.0);

  Kokkos::View<C*> primary("failclosed Bianchi primary", 1);
  Kokkos::View<C*> rhs("failclosed Bianchi rhs",
                       fixture.transport->state().size());
  Kokkos::deep_copy(primary, C(1.0, 0.0));
  Kokkos::deep_copy(rhs, C(9.0, -4.0));
  const auto state_view = fixture.transport->state();
  const transport_type::const_flat_view state(state_view.data(),
                                               state_view.size());
  const transport_type::flat_view output(rhs.data(), rhs.size());
  auto incomplete_capability = fixture.capability();
  incomplete_capability.radial_boundary_treatment_qualified = false;
  int producer_calls = 0;
  bool stage_rejected = false;
  try {
    fixture.transport->evaluate_stage(
        fixture.execution, 0.0, primary, state, output,
        fixture.transport->next_generation(), incomplete_capability,
        [&](const auto&, double, const auto&, const auto) { ++producer_calls; });
  } catch (const std::invalid_argument&) {
    stage_rejected = true;
  }
  CHECK(stage_rejected);
  CHECK(producer_calls == 0);

  auto mismatched_boundary = fixture.capability();
  mismatched_boundary.boundary_evidence_id = "different-boundary-evidence";
  stage_rejected = false;
  try {
    fixture.transport->evaluate_stage(
        fixture.execution, 0.0, primary, state, output,
        fixture.transport->next_generation(), mismatched_boundary,
        [&](const auto&, double, const auto&, const auto) { ++producer_calls; });
  } catch (const std::invalid_argument&) {
    stage_rejected = true;
  }
  CHECK(stage_rejected);
  CHECK(producer_calls == 0);

  fixture.transport->evaluate_stage(
      fixture.execution, 0.0, primary, state, output,
      fixture.transport->next_generation(), fixture.capability(),
      complete_primary_producer(1.0, true));
  fixture.execution.fence("finish failclosed Bianchi stage");
  const auto host_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                            rhs);
  const auto readiness = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.transport->readiness());
  const auto stage = fixture.transport->latest_stage();
  const auto fields = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          stage.fields);
  const auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          stage.stamps);
  CHECK(readiness(0) == 0);
  for (std::size_t i = 0; i < host_rhs.size(); ++i) {
    CHECK_COMPLEX_NEAR(host_rhs(i), C{}, 0.0);
  }
  for (std::size_t i = 0; i < fields.size(); ++i) {
    CHECK_COMPLEX_NEAR(fields.data()[i], C{}, 0.0);
    CHECK(stamps.data()[i] == 0);
  }
}

TEST_CASE("plus2 Bianchi transport rejects nonphysical Kerr parameters") {
  TransportFixture fixture;
  const auto rejects = [&](const teuk::KerrParameters& parameters) {
    bool rejected = false;
    try {
      transport_type invalid(
          fixture.execution, fixture.registry, fixture.grid, parameters,
          ell_max, fixture.cos_theta, fixture.sin_theta,
          teuk::RadialDiscretization::D105, "invalid_Kerr_transport");
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    return rejected;
  };
  CHECK(rejects({std::numeric_limits<double>::quiet_NaN(), 0.0, 1.3}));
  CHECK(rejects({1.0, std::numeric_limits<double>::infinity(), 1.3}));
  CHECK(rejects({1.0, 1.01, 1.3}));
  CHECK(rejects({1.0, 0.9, std::numeric_limits<double>::quiet_NaN()}));
}

TEST_CASE("plus2 Bianchi hot stage allocates and fences only at setup seams") {
  TransportFixture fixture;
  Kokkos::View<C*> primary("no-fence Bianchi primary", 1);
  Kokkos::deep_copy(primary, C(1.0, 0.0));
  teuk::DeviceRK4Workspace<C, execution_space> workspace(1);
  auto producer = complete_primary_producer();
  auto observer = [](const auto&, double, std::uint64_t, const auto&,
                     const auto&) {};
  teuk::device_one_way_bianchi_transport_rk4_step(
      fixture.execution, primary, 0.0, 0.01, primary_rhs(), producer,
      observer, workspace, *fixture.transport, fixture.capability());
  fixture.execution.fence("warm Bianchi transport kernels");

  bianchi_transport_allocations = 0;
  bianchi_transport_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_bianchi_transport_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_bianchi_transport_fence);
  teuk::device_one_way_bianchi_transport_rk4_step(
      fixture.execution, primary, 0.01, 0.01, primary_rhs(), producer,
      observer, workspace, *fixture.transport, fixture.capability());
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish no-allocation Bianchi stage");
  CHECK(bianchi_transport_allocations == 0);
  CHECK(bianchi_transport_fences == 0);
}
