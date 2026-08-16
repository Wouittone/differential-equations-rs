# AutoDP5 regular-ODE handoff

## Scope and upstream reference

This wave ports the regular first-order ODE surface of
`OrdinaryDiffEqLowOrderRK.AutoDP5` at pinned revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. The upstream constructor is
defined at `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:218` as
`AutoAlgSwitch(DP5(), alg; kwargs...)`; the DP5 component is the adaptive
Dormand--Prince 5/4 algorithm at lines 200--216. The upstream composite
switch state and stiffness detector live in
`lib/OrdinaryDiffEqCore/src/composite_algs.jl:1-140`.

## Rust surface

`AutoDp5<A>::new(stiff_algorithm)` and the uppercase compatibility alias
`AutoDP5<A>` retain the configured stiff component and delegate regular ODE
stepping to the existing pinned `Dp5` kernel. This preserves fixed and
adaptive stepping, DP5 error control, forward/backward integration, callbacks,
`save_at`, dense output, and allocation behavior without duplicating the
tableau or perform-step implementation. The public export is in `src/lib.rs`;
focused behavior coverage is in `tests/autodp5.rs` and the low-order Julia
fixture imports `AutoDP5(Rodas5P())`.

## Explicit limitation

The Rust driver currently has no native stiffness detector or in-flight
algorithm-switch state. Consequently, this facade does not switch to the
configured stiff component; it is a regular-ODE compatibility mapping to the
nonstiff DP5 component. Automatic stiffness switching remains deferred and
must not be represented as silently calling an external wrapper or as DAE-only
behavior. A future switch implementation must preserve this constructor and
replace only the delegation path with native `AutoAlgSwitch` state.

## Evidence

- `cargo fmt --all -- --check` passed.
- `cargo test --all-targets` passed.
- `cargo clippy --all-targets -- -D warnings` passed in the isolated worker.
- `git diff --check` passed.
- Julia pinned/full validation was completed in the isolated worker before
  integration; the coordinator remains blocked by `JULIA-PATH-20260809`.
