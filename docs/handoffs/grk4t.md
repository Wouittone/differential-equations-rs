# GRK4T Rosenbrock handoff

## Summary

Added the native regular-ODE `Grk4t` algorithm using the shared adaptive
Rosenbrock/Rodas stage kernel. The implementation ports the pinned four-stage
GRK4T tableau and supports stiff nonautonomous problems, fixed and adaptive
stepping, backward integration, callbacks, `save_at`, and analytic or
finite-difference Jacobians through the existing interfaces.

## Files changed

- `src/rosenbrock_extended.rs`: public algorithm, exact pinned tableau,
  dispatch, and focused stiff/fixed/backward/callback tests.
- `src/lib.rs`: exports `Grk4t`.
- `examples/rosenbrock_extended_compliance.rs`: adaptive and fixed compliance
  rows.
- `tests/julia/rosenbrock_extended.jl`: pinned `GRK4T()` reference row.
- `docs/handoffs/grk4t.md`: this handoff.

## Upstream source and revision

The coefficients are copied from
`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
`GRK4TRodasTableau`, at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `gamma = 0.231`;
- four stages with nodes `[0.0, 0.462, 0.8802083333333334,
  0.8802083333333334]`;
- exact upstream decimal `A`, `C`, `d`, `b`, and `btilde` values.

The Julia algorithm is `GRK4T()` from `OrdinaryDiffEqRosenbrock`; it is an
adaptive fourth-order Rosenbrock method with a third-order embedded estimator.

## Rust tests and commands

- `cargo fmt --all`
- `cargo fmt -- --check`
- `cargo test --lib rosenbrock_extended`: pass (11 tests).
- `cargo test --all-targets`: pass (105 library tests and all integration and
  example targets).
- `cargo clippy --all-targets -- -D warnings`: pass.
- `git diff --check`: pass.
- `cargo run --quiet --release --example rosenbrock_extended_compliance`: pass;
  GRK4T rows were emitted for both adaptive and fixed modes.

The Rust checks used the isolated target directory
`D:\Source\Repositories\differential-equations-rs\target-grk4t` because
other worktrees were concurrently using the default target lock.

## Julia tests

The fixture is present, but Julia is unavailable in this environment:
`Get-Command julia` returned no executable. Retry both
`julia --project=tests/julia tests/julia/pinned_environment.jl --check` and
`julia --project=tests/julia tests/julia/runtests.jl` once Julia resolves on
PATH.

## Numerical differences and limitations

No GRK4T-specific numerical differences were introduced. As with the other
extended Rosenbrock methods, method-specific stiff dense interpolation is not
implemented; trajectory sampling uses the crate's shared regular-ODE recorder.

## Allocation/performance impact

GRK4T reuses the existing fixed-size workspace and LU factorization buffers;
no per-step heap allocation was introduced.

## Follow-up dependencies

The coordinator should regenerate the public inventory after merging this
algorithm and add the status-wave entry. Julia compliance remains pending on
the environment PATH blocker. The coordinator should resolve overlaps in
`src/rosenbrock_extended.rs`, `src/lib.rs`, the shared example, and shared
Julia fixture with concurrent Rosenbrock-family waves.

