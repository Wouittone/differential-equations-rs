# ROS3PRL regular ODE handoff

Added the native `Ros3Prl` solver for OrdinaryDiffEqRosenbrock's `ROS3PRL`
algorithm. It is a four-stage, third-order, stiffly accurate low-storage
Rosenbrock method with the pinned second-order embedded estimator. The solver
uses the existing allocation-free Rosenbrock/Rodas driver, including analytic
or finite-difference Jacobians, reusable dense factorization, adaptive control,
callbacks, reverse-time integration, and shared `save_at` recording.

## Pinned upstream references

The source revision is `211142263781255a9aa2f910f6760b9f18ec29c8`.

- `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:868-894`:
  `ROS3PRLRodasTableau`, including the four-stage `A`, `C`, `c`, `d`, `b`, and
  `btilde` coefficients.
- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:256-261`: ROS3PRL metadata,
  including primary order 3 and the regular ODE classification.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_caches.jl:448-468`:
  `tabtype(::ROS3PRL)` and membership in the regular Rosenbrock tableau cache.
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:729-861`:
  the shared regular ODE Rosenbrock stage, update, and embedded-error path.
- `lib/OrdinaryDiffEqRosenbrockTableaus/test/ode_rosenbrock_tests.jl:500-517`:
  pinned ROS3PRL linear and two-dimensional convergence/success coverage.

No DAE, SDE, wrapper, or method-specific stiff interpolation behavior is
included; regular ODE trajectory samples use the crate's shared recorder.

## Rust coverage

- `src/rosenbrock_extended.rs`: public constructor, exact Float64 tableau,
  fixed/adaptive/backward integration, analytic Jacobian, callback invalidation,
  `save_at`, and fixed-step order/lifecycle tests.
- `src/lib.rs`: public `Ros3Prl` export.
- `tests/rosenbrock_driver.rs`: callback-free allocation invariant.
- `examples/rosenbrock_extended_compliance.rs` and
  `tests/julia/rosenbrock_extended.jl`: fixed/adaptive endpoint compliance
  row against upstream `ROS3PRL()`.

## Verification

Passed on the isolated `codex/rosenbrock-ros3prl` worktree:

- `cargo fmt --all`
- focused ROS3PRL lifecycle and order tests
- `cargo test --test rosenbrock_driver -- --nocapture`
- `cargo run --quiet --example rosenbrock_extended_compliance`
- `git diff --check`

- `cargo test --all-targets` — passed (all tests).
- `cargo clippy --all-targets -- -D warnings` — passed.

Julia is not installed on PATH in this environment (`julia --version` is
unavailable).
Retry with the exact pinned project command:

```text
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/rosenbrock_extended.jl
```
