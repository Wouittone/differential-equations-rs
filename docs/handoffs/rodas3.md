# Rodas3 handoff

## Summary

Added the native regular-ODE `Rodas3` algorithm using the shared Rosenbrock
driver and the exact four-stage tableau from the pinned upstream revision.

## Files changed

- `src/rosenbrock_extended.rs`
- `src/lib.rs`
- `examples/rosenbrock_extended_compliance.rs`
- `tests/rosenbrock_driver.rs`
- `tests/julia/rosenbrock_extended.jl`

## Public API

- `differential_equations::Rodas3`

## Upstream source and revision

`D:/Source/_review/OrdinaryDiffEq.jl/lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`, revision
`211142263781255a9aa2f910f6760b9f18ec29c8`, function
`Rodas3RodasTableau(::Type{T}, ::Type{T2})`.

The port preserves `gamma = 1/2`, stage coefficients `A` and `C`, nodes
`[0, 0, 1, 1]`, time-derivative weights `[1/2, 3/2, 0, 0]`, solution weights
`[2, 0, 1, 1]`, and the embedded estimator `[0, 0, 0, 1]`. The upstream dense
matrix is empty, so sampled output uses the shared recorder just as for the
existing Rosenbrock tableau methods.

## Rust tests

- Stiff nonautonomous adaptive solve and fixed-step order-three convergence.
- Backward integration, callbacks, and `save_at` behavior.
- Analytic and finite-difference Jacobian agreement and RHS-work reduction.
- Callback-free allocation invariance in `tests/rosenbrock_driver.rs`.

## Julia tests

Added `Rodas3()` to `tests/julia/rosenbrock_extended.jl`, including adaptive
stiff and fixed exponential compliance rows. Julia was not available in this
environment (`Get-Command julia` returned no command); retry the pinned
environment check and full Julia suite when `julia` resolves on PATH.

## Commands run

- `cargo fmt --all`
- `cargo test --all-targets` (104 unit tests and all integration/example targets passed before the allocation-test addition)
- `cargo test --test rosenbrock_driver` (passed)
- `cargo clippy --all-targets -- -D warnings` (passed with `CARGO_TARGET_DIR` redirected to avoid a concurrent shared target lock)
- `git diff --check` (passed)
- `cargo run --example rosenbrock_extended_compliance` (passed; emitted Rodas3 adaptive/fixed rows)

## Numerical differences

No intentional coefficient or stage-equation differences. Adaptive step
controller details are shared with the existing Rust Rosenbrock family and may
choose a different accepted-step sequence from Julia.

## Allocation/performance impact

Rodas3 reuses the existing fixed-size workspace and factorization cache. The
allocation regression test passed with one-step and thousand-step solves.

## Known limitations

The shared recorder currently supplies trajectory sampling instead of the
method-specific dense interpolant. Julia compliance remains pending the local
Julia executable blocker.

## Follow-up dependencies

Coordinator should regenerate the public ODE inventory after cherry-picking
this commit and update the overnight status. Run the pinned and full Julia
gates when Julia is available.
