# Rodas4P2 handoff

## Summary:

Implemented the native regular-ODE `Rodas4P2` Rosenbrock/Rodas algorithm using
the exact pinned tableau from OrdinaryDiffEq. The method is exposed as a
zero-sized Rust algorithm and runs through the shared Rosenbrock driver for
fixed and adaptive stepping, analytic or finite-difference Jacobians,
backward integration, callbacks, `save_at`, and endpoint recording.

## Files changed:

- `src/rosenbrock_extended.rs`
- `src/lib.rs`
- `examples/rosenbrock_extended_compliance.rs`
- `tests/rodas4p2.rs`
- `tests/rodas4p2_allocations.rs`
- `tests/julia/rosenbrock_extended.jl`

## Public APIs added:

- `differential_equations::Rodas4P2`

## Upstream source and revision:

- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
  `Rodas4P2Tableau`
- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl`, `:Rodas4P2`
- Revision `211142263781255a9aa2f910f6760b9f18ec29c8`

The six-stage `A`, `C`, `c`, `d`, `b`, and `btilde` data are copied exactly
from the pinned source. The two-row `H` stiff interpolation matrix is not
represented by the current Rust tableau type; regular-ODE sampling uses the
shared accepted-segment recorder, as with the existing native Rodas methods.

## Rust tests:

- Fixed-step fourth-order convergence on exponential growth.
- Backward integration.
- Adaptive stiff nonautonomous solve with an analytic Jacobian.
- Continuing callback and `save_at` samples.
- Callback-free allocation invariance across one and 1000 fixed steps.

## Julia tests:

Added `Rodas4P2()` to the pinned extended Rosenbrock fixture. Julia execution
was not available in this worker (`julia` is not recognized); retry with:

```powershell
julia --project=tests/julia tests/julia/runtests.jl
```

## Commands run:

- `cargo fmt --all`
- `cargo fmt -- --check`
- `cargo test --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`
- `cargo run --quiet --release --example rosenbrock_extended_compliance`

All commands passed. The compliance example reported finite adaptive and
fixed endpoints (`5.403023073098626e-1` and `2.718281828436041e0`).

## Numerical differences:

The shared recorder uses its accepted-segment interpolation for regular-ODE
`save_at` values rather than OrdinaryDiffEq's method-specific `H` interpolant.
Endpoint fixed-step results use the pinned tableau directly.

## Allocation/performance impact:

The method reuses the existing allocation-free Rosenbrock workspace. The
dedicated allocation test confirms one-step and 1000-step callback-free solves
have identical allocation counts.

## Known limitations:

DAE-only residual behavior, stiff-specific `H` dense interpolation, and
external Julia wrappers are outside this regular-ODE port.

## Follow-up dependencies:

Coordinator should regenerate the machine-readable inventory and update the
overnight status counts after exporting the new public constructor.

## Recommended next task:

Port the next missing regular-ODE Rosenbrock/Rodas constructor, preserving the
shared tableau and driver interfaces.
