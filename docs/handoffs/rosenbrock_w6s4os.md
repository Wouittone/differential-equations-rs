# RosenbrockW6S4OS handoff

## Summary

Added the native regular ODE `RosenbrockW6S4OS` fixed-step Rosenbrock-W
algorithm. It uses the six-stage fourth-order tableau published by the pinned
`OrdinaryDiffEqRosenbrockTableaus` source and the existing dense LU/Jacobian
kernel used by the Rosenbrock/Rodas family.

## Files changed

- `src/rosenbrock_extended.rs`: public algorithm, exact six-stage A/C/c/d/b
  tableau, fixed-step capability, convergence and backward/unsupported-mode
  tests.
- `src/lib.rs`: public re-export.
- `examples/rosenbrock_extended_compliance.rs`: fixed-step endpoint row.
- `tests/julia/rosenbrock_extended.jl`: pinned Julia fixed-step comparison.

## Public API

`differential_equations::RosenbrockW6S4OS` implements `OdeAlgorithm` and
requires `SolveOptions { adaptive: false, initial_step: Some(...) }`.

## Upstream source and revision

- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:150-184`
- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:923-978`
- pinned revision `211142263781255a9aa2f910f6760b9f18ec29c8`

The upstream method has no embedded `btilde` estimator and is documented as
fixed-step-only. The Rust tableau preserves its zero error-weight vector and
all published coefficients, including the nonzero final stage weight.

## Rust tests

The module tests cover fixed-only rejection, fourth-order fixed convergence,
backward integration, callbacks/save-at through the shared driver, and the
existing analytic-vs-finite-difference Jacobian path. The all-target suite
passed with 98 library tests plus all integration tests.

## Julia tests

`tests/julia/rosenbrock_extended.jl` now compares the fixed-step endpoint of
`RosenbrockW6S4OS()` against the Rust compliance row. Julia could not be
executed in this environment because `julia` is absent from PATH; retry with:

```powershell
Get-Command julia
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

## Commands run

- `cargo fmt --all` (pass; `cargo fmt -- --check` should be rerun by the coordinator)
- `cargo test --all-targets --target-dir target-w6s4os` (pass)
- `cargo run --quiet --release --example rosenbrock_extended_compliance` (pass)
- `git diff --check` (pending coordinator merge)
- Julia gates unavailable: executable not on PATH.

## Numerical differences and limitations

The shared Rust driver intentionally rejects adaptive scheduling for this
method, matching upstream's fixed-step-only contract. Dense output uses the
shared accepted-segment interpolation because the pinned tableau has no H
dense matrix. Mass-matrix/DAE-only behavior and external linsolve/autodiff
configuration are out of regular `OdeProblem` scope. The existing family
kernel invalidates the Jacobian cache after accepted steps, while preserving
reuse across rejected attempts; this is the current shared-kernel policy and
does not change the W-method's numerical result.

The compliance example emitted the fixed endpoint
`2.71828182843189037e0` for `u'=u`, `dt=0.01`, which is within the method's
fourth-order fixed-step error envelope.

## Allocation/performance impact

The method reuses the existing max-eight-stage workspace and dense LU buffers;
no per-step heap allocation was introduced. The existing Rosenbrock allocation
invariance integration test passed.

## Follow-up dependencies

The coordinator should regenerate the public ODE inventory after merging and
run the full pinned Julia suite when `julia` resolves on PATH.

## Recommended next task

Continue with another independent native Rosenbrock/Rodas tableau family while
keeping `src/rosenbrock_extended.rs` ownership serialized.
