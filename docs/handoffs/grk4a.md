# GRK4A Rosenbrock handoff

## Summary

Added the native regular-ODE `Grk4a` algorithm. It uses the existing shared
Rosenbrock/Rodas stage driver, dense LU factorization, finite-difference or
analytic Jacobian provider, time-derivative estimate, adaptive controller, and
callback lifecycle.

## Files changed

- `src/rosenbrock_extended.rs`: public algorithm, pinned four-stage tableau,
  focused stiff/fixed/backward/callback/Jacobian tests.
- `src/lib.rs`: exports `Grk4a`.
- `examples/rosenbrock_extended_compliance.rs`: adaptive and fixed compliance
  rows.
- `tests/julia/rosenbrock_extended.jl`: pinned `GRK4A()` comparison.
- `docs/handoffs/grk4a.md`: this handoff.

## Upstream source and revision

The coefficients are copied from
`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
`GRK4ARodasTableau`, at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `gamma = 0.395`;
- four stages with nodes `[0.0, 0.438, 0.87, 0.87]`;
- exact upstream decimal `A`, `C`, `d`, `b`, and `btilde` values.

The Julia algorithm is `GRK4A()` from `OrdinaryDiffEqRosenbrock` and is an
adaptive fourth-order Rosenbrock method with a third-order embedded estimator.

## Rust tests

- Stiff nonautonomous adaptive endpoint reaches `cos(1)`.
- Fixed-step exponential convergence is fourth order.
- Forward/backward adaptive integration works.
- Continuous callbacks and `save_at` are honored and invalidate stiff caches.
- Analytic and finite-difference Jacobians agree; the analytic path reduces
  RHS work.
- The shared callback-free allocation regression includes GRK4A and confirms
  allocation count is invariant from one to 1,000 fixed steps.

## Commands run

- `cargo fmt --all`
- `cargo fmt -- --check`
- `cargo test --lib rosenbrock_extended`
- `cargo test --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`
- `cargo run --quiet --release --example rosenbrock_extended_compliance`

All Rust commands passed. The worktree target directory was concurrently
locked, so verification used the dedicated target directory
`D:\Source\Repositories\differential-equations-rs\target-grk4a`.

## Julia tests

The Julia fixture is present, but `julia --project=tests/julia
tests/julia/pinned_environment.jl --check` could not run because the
coordinator environment has no `julia` executable on PATH. Retry both pinned
environment and full suite once `Get-Command julia` resolves an executable.

## Numerical differences and limitations

No GRK4A-specific numerical differences were introduced. As with the existing
extended Rosenbrock methods, stiff method-specific dense interpolation is not
implemented; trajectory sampling uses the shared regular-ODE recorder.

## Allocation/performance impact

GRK4A reuses the existing fixed-size workspace and factorization buffers; no
per-step heap allocation was introduced.

## Follow-up dependencies

The coordinator must regenerate the public ODE inventory after merging this
algorithm and add the status-wave entry. Julia compliance remains pending on
the environment PATH blocker.

## Recommended next task

Port another independent native Rosenbrock/Rodas tableau while serializing
ownership of `src/rosenbrock_extended.rs`.
