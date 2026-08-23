# Rodas3d handoff

## Summary

Added the native regular-ODE `Rodas3d` algorithm using the shared Rosenbrock
driver and the exact four-stage tableau from the pinned upstream revision.

## Public API and files

- `differential_equations::algorithms::rosenbrock::Rodas3d` is publicly exported.
- Tableau and perform-step wiring are in `src/rosenbrock_extended.rs`.
- Compliance output is in `examples/rosenbrock_extended_compliance.rs` and
  `tests/julia/rosenbrock_extended.jl`.
- Allocation coverage is in `tests/rosenbrock_driver.rs`.

## Upstream source and revision

`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`, revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, function
`Rodas3dRodasTableau(::Type{T}, ::Type{T2})`.

The port preserves `gamma = 0.57281606`, stage coefficients `A` and `C`,
nodes `[0, 1.2451051999132263, 1, 1]`, time-derivative weights
`[0.57281606, -3.819703409768521, 0, 0]`, solution weights
`[1.745761108723104, 0, 1, 1]`, and embedded estimator `[0, 0, 0, 1]`.
The upstream dense matrix is empty, so sampled output uses the shared
accepted-step recorder.

## Coverage

Rust tests cover adaptive stiff integration, fixed-step linear order (the
method is fourth order on linear problems for this damping root), nonlinear
shared-driver behavior, backward integration, callbacks, `save_at`, analytic
versus finite-difference Jacobians, rejected steps, and allocation invariance.

The compliance example emits both `rodas3d_adaptive` and `rodas3d_fixed` rows.
Julia fixture coverage includes `Rodas3d()` for both paths.

## Verification

- `cargo fmt --all`
- `cargo test --lib` (109 passed)
- `cargo run --quiet --example rosenbrock_extended_compliance` (passed)

The full `cargo test --all-targets`, `cargo clippy --all-targets -- -D
warnings`, `cargo fmt -- --check`, and `git diff --check` gates are pending
final handoff execution. Julia was not available in this environment; retry
the pinned/full Julia fixture when `julia` resolves on PATH.
