# Generated Vern6 coefficient wave

The existing public `Vern6` constructor is retained over the generic explicit
RK kernel, while its exact pinned tableau is moved into the deterministic
generated coefficient fixture.

Pinned source: OrdinaryDiffEqVerner at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, specifically
`lib/OrdinaryDiffEqVerner/src/vern6_tableaus.jl` and the Vern6 perform-step
path in `lib/OrdinaryDiffEqVerner/src/vern6_perform_step.jl`. The Rust fixture
preserves all nine stage nodes, strictly lower-triangular coefficients,
sixth-order weights, and embedded defect weights. Method-specific lazy dense
stages remain outside the generic driver, as before.

`tests/verner6.rs` adds fixed convergence and adaptive/backward/callback
regressions. Existing `tests/julia/verner.jl` already compares Vern6 fixed and
adaptive endpoints against the pinned Julia implementation; no duplicate
fixture is needed. No DAE, split, SDE, wrapper, or external behavior is added.
