# ROS3P handoff

## Summary

Added the native regular-ODE `Ros3p` algorithm, porting the pinned upstream
`ROS3PRodasTableau` through the existing shared Rosenbrock/Rodas kernel. The
method has three stages, third-order primary weights, and the upstream
second-order embedded estimator.

## Files changed

- `src/rosenbrock_extended.rs`: `Ros3p`, exact Float64 tableau, dispatch, and
  stiff/nonautonomous, fixed-order, callback/save, Jacobian, and backward tests.
- `src/lib.rs`: public `Ros3p` export.
- `examples/rosenbrock_extended_compliance.rs`: Rust compliance output row.
- `tests/julia/rosenbrock_extended.jl`: `ROS3P()` reference row.

## Public API

`differential_equations::Ros3p` implements `OdeAlgorithm` and supports the
shared regular `OdeProblem` interface, finite-difference or analytic Jacobians,
adaptive and fixed stepping, callbacks, backward integration, and `save_at`.

## Upstream source and revision

- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl`, `ROS3P` entry
- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
  `ROS3PRodasTableau`
- Revision `211142263781255a9aa2f910f6760b9f18ec29c8`

## Verification

- `cargo fmt -- --check`: pass.
- `cargo test --all-targets`: pass (104 unit tests plus integration/example
  targets), using isolated `CARGO_TARGET_DIR` because the shared target lock
  was held by another worktree.
- `cargo clippy --all-targets -- -D warnings`: pass using isolated target.
- `git diff --check`: pass.
- Julia was unavailable: `Get-Command julia` returned no executable. Retry:
  `julia --project=tests/julia tests/julia/pinned_environment.jl --check`, then
  `julia --project=tests/julia tests/julia/runtests.jl` once Julia resolves on
  PATH.

## Numerical differences and limitations

The dynamic Julia expressions are represented by their pinned Float64 values;
this keeps the hot path static and allocation-free. The generic crate recorder
is used for trajectory sampling rather than the method-specific upstream dense
interpolant. Adaptive backward integration of this A-stable method on a
physically unstable reversed-time scalar decay can take overly large accepted
steps; the focused backward test therefore uses fixed `dt=0.01`, while the
adaptive path is covered on stiff nonautonomous forward integration and the
Rust/Julia compliance fixture covers both modes forward.

## Allocation/performance impact

No per-step heap allocation was introduced; ROS3P uses the existing reusable
Rosenbrock workspace and LU factorization buffers.

## Follow-up

Coordinator should regenerate the inventory after merging and resolve the
expected `src/lib.rs`, extended module, example, and Julia fixture overlaps
with concurrent Rosenbrock-family waves. Run the pinned and full Julia suites
when Julia is on PATH.

