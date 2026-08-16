# ROS2 regular ODE handoff

Added the native `Ros2` implementation for upstream `ROS2` from the pinned
`OrdinaryDiffEqRosenbrock`/`OrdinaryDiffEqRosenbrockTableaus` revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

The implementation reuses the shared Rosenbrock/Rodas driver and cache. Its
two-stage tableau is copied from `ROS2RodasTableau`: `gamma =
1.7071067811865475`, stage coefficients `A[2,1] = 0.585786437626905` and
`C[2,1] = -1.17157287525381`, nodes `[0, 1]`, time-derivative weights
`[1.7071067811865475, -1.7071067811865475]`, solution weights
`[0.8786796564403574, 0.2928932188134525]`, and embedded `btilde`
`[0.2928932188134525, 0.2928932188134525]`. `Ros2` is the idiomatic Rust
spelling of upstream `ROS2`; inventory normalization maps the names together.

The generic Rosenbrock lifecycle provides finite-difference or analytic
Jacobians, time-derivative differentiation, one factorization per attempt,
adaptive order-2 controller, rejected-step cache reuse, backward integration,
callbacks, requested samples, and allocation-free repeated stepping. Focused
tests cover stiff nonautonomous and exponential convergence, backward solves,
callbacks/save-at, analytic-vs-finite-difference Jacobians, and allocation
invariance. `tests/julia/rosenbrock_extended.jl` compares adaptive and fixed
endpoints against `ROS2()`.

## Verification

- `cargo fmt -- --check`: pass
- `cargo test --all-targets`: pass
- `cargo clippy --all-targets -- -D warnings`: pass
- `git diff --check`: pass
- Julia checks: rerun from an environment where `julia` resolves on PATH;
  coordinator retry command is
  `julia --project=tests/julia tests/julia/pinned_environment.jl --check`
  followed by `julia --project=tests/julia tests/julia/runtests.jl`.
