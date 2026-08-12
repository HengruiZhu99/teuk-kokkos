# Complete-graph fourth-order gap

## Verdict

The D8-4 operator and common-stage RK4 do not yet make the complete 13-field
`SpatialPipeline` fourth order in the radial maximum norm.  A fixed-space
trajectory does converge at fourth order in time, and every direct
first-radial-derivative sector is fourth order at both endpoints.  The
quadratic source path, however, applies the radial operator to data which
already contain a numerical radial derivative.  Its endpoint error is third
order.  This is a production qualification blocker, not a reason to weaken the
fourth-order gate.

The executable evidence is
`tests/test_full_graph_convergence.cpp`.  It uses one stored and sharp-closed
`m=0` mode, `ell_max_first=3`, `ell_max_second=6`, and seven Gauss-Legendre
nodes.  Each initial field is synthesized from retained spin-weighted modes,
and the seven-node grid satisfies the exact-product padding condition.  The
radial profiles are non-polynomial and complex.  The nested 49, 97, and 193
point grids share both endpoints and every coarser coordinate.

## Measured radial orders

The table reports the 97-to-193 Richardson ratio and
`p=log2(ratio)`.  RMS includes every radial and angular point.  Endpoint is the
maximum over scri and the horizon-side boundary.

| graph quantity | RMS ratio | RMS p | endpoint ratio | endpoint p |
| --- | ---: | ---: | ---: | ---: |
| first Teukolsky triple | 20.3547 | 4.347 | 13.2818 | 3.731 |
| seven reconstruction fields | 21.8069 | 4.446 | 16.3398 | 4.031 |
| second Teukolsky triple | 20.3068 | 4.344 | 13.2273 | 3.725 |
| final coordinate forcing | 12.0080 | 3.586 | 7.42038 | 2.892 |
| inner-source value | 21.3688 | 4.416 | 16.3290 | 4.029 |
| inner-source tangent | 19.4637 | 4.283 | 7.71681 | 2.948 |
| independent reconstruction constraints | 22.4987 | 4.492 | 16.3347 | 4.030 |

The source endpoint ratios approach eight, not sixteen.  The forcing RMS
order is about 3.5 because a fixed number of third-order boundary points are
diluted by the growing number of higher-order interior points.  Consequently,
an RMS-only fit would obscure the maximum-norm failure.

## Exact derivative depth and call chain

The current free-damped production path has maximum radial derivative depth
two:

1. `SpatialPipeline::evaluate_rhs_at_time` calls
   `evaluate_sbp_teukolsky_full_stage_rhs` for the first triple.  Its scratch
   construction differentiates `Psi`, `Q`, and the reconstructed `Psi`
   velocity once.
2. `evaluate_reconstruction_chain(stage, output, ...)` differentiates each of
   the seven metric fields once and forms their first time derivatives.
3. The first-triple RHS and reconstruction RHS are then used as a new stage by
   a second Teukolsky call and a second reconstruction-chain call.  Those calls
   form the common-stage tangents and contain second radial derivatives of the
   original state.
4. `prepare_source_inputs` passes the first derivatives through the value
   slots and the second derivatives through `ddt`, `tangent_scratch`, and
   `reconstruction_tangent_radial_` into the derivative-tangent slots.
5. `evaluate_spatial_inner_source_tangent` combines these slots algebraically.
   The inner-source value has depth one; its tangent has depth two.
6. `project_inner_source` calls
   `evaluate_spatial_outer_source_from_ethprime`.  That kernel differentiates
   the projected `D` value once.  Because `D` already depends on depth-one
   primitives, `radial_D` has depth two.  It is combined algebraically with
   the depth-two inner-source tangent to form the coordinate forcing.
7. The second Teukolsky RHS adds the forcing algebraically; it does not
   differentiate it.  There is no depth-three radial path in the current
   graph.

The responsible production locations are the tangent graph and source input
assembly in `include/teuk/spatial_pipeline.hpp`, the first-derivative kernels
in `include/teuk/full_spatial.hpp`, and the outer `radial_D` operation in
`include/teuk/source_spatial.hpp`.

## Remedy order budget

A boundary-order-five first-derivative operator is sufficient in formal
truncation order for this graph.  If the first application has endpoint error
`O(h^5)`, the second application can lose one power through division by `h`
and still deliver `O(h^4)`.  Thus a qualified D10-5 SBP operator is a viable
remedy; boundary order six is not required by the current derivative depth.
This conclusion is conditional on measuring the composed graph: boundary
closure coefficients, error regularity across the closure rows, spectral
radius, compatible dissipation, and endpoint stability can still invalidate
the formal count.

An alternative is to retain D8-4 for first derivatives and implement direct
fourth-order endpoint formulas for the required second-derivative jets.  A
generic `D2` alone is not enough because the nested terms include products,
radially varying coefficients, and GHP `Delta_n` combinations.  The direct
route must expand those product and coefficient derivatives consistently, or
construct all first/second radial jets from the original RK stage before the
source algebra.  Either remedy requires new independent point or manufactured
oracles and the same endpoint-inclusive full-graph regression.

## Temporal result and remaining angular scope

At fixed 17-point D8-4 space, the complete common-stage graph was evolved with
16, 32, and 64 RK4 steps against a 512-step reference.  The first triple,
seven-field reconstruction sector, and second triple each have both
successive temporal ratios above 12.  This separates the spatial blocker from
the RK4 integrator: the time composition is fourth order once the spatial
operator is held fixed.

The manufactured radial test removes ordinary input-band and quadratic
product aliasing. A later rotating-Kerr source-only qualification, documented
in `PLUS2_FULL_SOURCE_ANGULAR_QUALIFICATION.md`, independently refines
`ell_max` and `N_theta` through the complete concrete primitive, ordered-pair,
projection, and outer-derivative graph. It closes the manufactured angular
discretization gap for that stage-local graph, but it is not a production
waveform campaign. Rational Kerr coefficients generate off-band content, so a
scientific run still needs observable-based `ell_max` and `N_theta` refinement.
Any selected angular filter must be included in that future study; the current
graph has no filter.
