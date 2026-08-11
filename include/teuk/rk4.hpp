#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace teuk {

// Preallocated classical-RK4 storage for a flat coupled state.  A component
// can be a scalar, complex number, or a small algebraic state supporting the
// operations used below.  The RHS callable has signature
//   rhs(time, const std::vector<Value>& state, std::vector<Value>& derivative)
// and is invoked on the whole coupled state at each common stage.
template <class Value>
struct RK4Workspace {
  explicit RK4Workspace(const std::size_t component_count)
      : stage(component_count),
        k1(component_count),
        k2(component_count),
        k3(component_count),
        k4(component_count) {}

  [[nodiscard]] std::size_t size() const { return stage.size(); }

  std::vector<Value> stage;
  std::vector<Value> k1;
  std::vector<Value> k2;
  std::vector<Value> k3;
  std::vector<Value> k4;
};

template <class Value>
inline void rk4_form_stage(const std::vector<Value>& initial,
                           const std::vector<Value>& slope,
                           const double scaled_step,
                           std::vector<Value>& stage) {
  const std::size_t n = initial.size();
  if (slope.size() != n || stage.size() != n) {
    throw std::invalid_argument("RK4 stage buffers have inconsistent sizes");
  }
  for (std::size_t i = 0; i < n; ++i) {
    stage[i] = initial[i] + scaled_step * slope[i];
  }
}

template <class Value>
inline void rk4_accumulate(const double step,
                           const std::vector<Value>& k1,
                           const std::vector<Value>& k2,
                           const std::vector<Value>& k3,
                           const std::vector<Value>& k4,
                           std::vector<Value>& state) {
  const std::size_t n = state.size();
  if (k1.size() != n || k2.size() != n || k3.size() != n ||
      k4.size() != n) {
    throw std::invalid_argument("RK4 slope buffers have inconsistent sizes");
  }
  for (std::size_t i = 0; i < n; ++i) {
    state[i] += (step / 6.0) *
                (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
}

template <class Value, class RightHandSide>
void classical_rk4_step(std::vector<Value>& state, const double time,
                        const double step, RightHandSide&& rhs,
                        RK4Workspace<Value>& workspace) {
  if (workspace.size() != state.size()) {
    throw std::invalid_argument("RK4 workspace does not match state size");
  }

  rhs(time, state, workspace.k1);

  rk4_form_stage(state, workspace.k1, 0.5 * step, workspace.stage);
  rhs(time + 0.5 * step, workspace.stage, workspace.k2);

  rk4_form_stage(state, workspace.k2, 0.5 * step, workspace.stage);
  rhs(time + 0.5 * step, workspace.stage, workspace.k3);

  rk4_form_stage(state, workspace.k3, step, workspace.stage);
  rhs(time + step, workspace.stage, workspace.k4);

  rk4_accumulate(step, workspace.k1, workspace.k2, workspace.k3,
                 workspace.k4, state);
}

}  // namespace teuk
