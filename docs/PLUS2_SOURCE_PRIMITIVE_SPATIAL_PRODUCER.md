# Spin `+2` source primitive spatial producer

**Status:** standalone, allocation-free D10-5 producer for the source-owned
common-stage inputs; not runtime wiring or a physical Bianchi boundary-data
qualification.

`Plus2SourcePrimitiveSpatialProducer` consumes generation-stamped
reconstruction `h[0]`, `h[1]`, and `h[2]` views together with the read-only
Route-A transported-curvature and Bianchi-derivative stages. It constructs
exactly the producer-owned production slots:

- twelve non-curvature primitives: `H,Sig,Kap,Rh,Ta,Al,Be,Ep,Pi,V,C,B`;
- seven metric `J/K` derivative value/tangent pairs:
  `Delta2Csharp,ethprime1Bsharp,Delta2C,eth1B,Delta2V,eth2C,ethprime2Csharp`;
- three value-only `Q` derivatives:
  `Delta2Sig,Delta3Kap,ethprime3Kap`.

`Z0,Z1` and the four curvature derivative pairs
`Delta4Z1,ethprime4Z1,Delta5Z0,eth5Z0` remain owned by the Bianchi transport.
The producer neither writes nor stamps those six output slots. Composition
must copy them through the existing typed transport adapter; there is no
second authority and no curvature recomputation.

The formulas use the reviewed ordinary-NP primitive evaluator algebra, the
repository GHP angular operators, exact stage tangents, signed
`X_m^sharp=conj(X_-m)` lookup, and the selected D10-5 radial operator. Every
view and angular plan is allocated at construction. Repeated evaluation
launches no allocation, copy, or fence.

Because radial and angular transforms couple collocation points, validation
is global: one stale or nonfinite reconstruction, curvature, or Bianchi slot
invalidates the invocation. Producer-owned values are cleared and their
stamps are zero; transport-owned output rows remain untouched. Generation
must be nonzero, exactly match the reconstruction and target stages, and
increase between invocations.

The production `Q` contract is intentionally value-only. `Q_T` and the
per-family diagnostic tangents of `Delta2Sig`, `Delta3Kap`, and
`ethprime3Kap` require `h[3]` and are not produced or replaced by zeros.
Likewise, no local `Psi0/R^5` or `Psi1/R^4` quotient is formed, so the producer
does not guess, extrapolate, or zero a scri coefficient.

Focused serial evidence covers the twelve formulas and Jet tangents against
the reviewed point oracle, exact output ownership, signed sharp lookup,
stale/nonfinite fail closure, endpoint-inclusive D10-5 convergence of both a
connection primitive and the nested `Delta3Kap` Q slot, and zero hot-path
allocations/fences. This does not qualify the physical initial or radial
boundary prescription for the passive Bianchi state.
