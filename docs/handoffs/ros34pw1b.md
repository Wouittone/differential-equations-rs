# ROS34PW1b regular ODE handoff

Added the native regular-ODE `Ros34Pw1b` algorithm, the Rust spelling of
OrdinaryDiffEqRosenbrock's `ROS34PW1b`. The solver uses the shared
Rosenbrock/Rodas stage kernel, dense LU factorization, Jacobian provider (or
finite differences), adaptive controller, callbacks, and trajectory recorder.
It is an adaptive four-stage method with a fourth-order primary formula and a
third-order embedded estimator (`ERROR_ORDER = 3`), matching upstream's
`alg_order(ROS34PW1b) = 3`.

## Pinned upstream references

The port is pinned to OrdinaryDiffEq revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:780-807`
  (`ROS34PW1bRodasTableau`), including all `A`, `C`, `gamma`, `c`, `d`, `b`,
  and `btilde` values.
- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:232-236`
  (`ROS34PW1b` metadata and W-method classification).
- `lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl:17,49`
  (`alg_order = 3` and `isWmethod = true`).
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:418-728`, the
  generic `RodasTableau` stage path used by the shared Rust kernel.

## Rust surface and coverage

- `src/rosenbrock_extended.rs`: tableau, `Ros34Pw1b` constructor type, and
  shared-driver implementation.
- `src/lib.rs`: public `Ros34Pw1b` export.
- `examples/rosenbrock_extended_compliance.rs` and
  `tests/julia/rosenbrock_extended.jl`: adaptive and fixed endpoint rows.
- Focused tests cover fixed-step third-order convergence, adaptive stiff
  integration, bounded-step backward integration, analytic Jacobian use,
  callbacks, `save_at`, and callback-free allocation invariance.

The regular ODE port intentionally excludes upstream DAE, wrapper, and SDE
paths. Method-specific stiff dense interpolation is outside this crate's
shared recorder contract.

## Verification

- `cargo fmt -- --check`: pass
- `cargo test --all-targets`: pass (106 library tests and all integration/example targets)
- `cargo clippy --all-targets -- -D warnings`: pass
- `git diff --check`: pass
- `cargo run --release --example rosenbrock_extended_compliance`: pass; emits
  `ros34pw1b_adaptive` and `ros34pw1b_fixed` rows.
- Julia gate: not run because `julia` is not installed. Retry from the
  repository root with `julia --project=tests/julia
  tests/julia/pinned_environment.jl --check`, followed by `julia
  --project=tests/julia tests/julia/runtests.jl`, after installing Julia and
  instantiating the pinned OrdinaryDiffEq environment.
