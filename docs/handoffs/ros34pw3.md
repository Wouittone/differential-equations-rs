# ROS34PW3 regular ODE handoff

Added the native `Ros34Pw3` Rust algorithm for upstream `ROS34PW3`, a
four-stage, fourth-order Rosenbrock-W method with a third-order embedded
estimator. The implementation is regular ODE only: DAE, AMF, SDE, and other
wrappers are intentionally out of scope.

## Pinned upstream references

All upstream references are from OrdinaryDiffEq revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:244` (`ROS34PW3` metadata)
- `lib/OrdinaryDiffEqRosenbrock/src/alg_utils.jl:19` (primary order 4) and
  `:51` (W-method classification)
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl:242-265`,
  `ROS34PW3RodasTableau`
- `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl:729-862`,
  shared `RosenbrockCache` `perform_step!` stage/update/error path

The Rust tableau copies the pinned `gamma`, `A`, `C`, `c`, `d`, `b`, and
`btilde` values exactly. The shared native Rosenbrock driver handles analytic
or finite-difference Jacobians, reused factorizations, adaptive and fixed
scheduling, backward spans, callbacks, `save_at`, and endpoint recording.

## Changes

- `src/rosenbrock_extended.rs`: public `Ros34Pw3`, exact pinned tableau, and
  shared-driver registration.
- `src/lib.rs`: public `Ros34Pw3` export.
- `tests/ros34pw3_driver.rs`: fourth-order convergence, backward integration,
  adaptive Jacobian/callback/`save_at`, and callback-free allocation coverage.
- `examples/ros34pw3_compliance.rs`: adaptive and fixed endpoint rows.
- `tests/julia/ros34pw3.jl`: pinned Julia compliance fixture.

## Verification

Passed locally:

- `cargo test --test ros34pw3_driver`
- `cargo test --lib rosenbrock_extended::tests::methods_have_their_expected_fixed_step_orders`
- `cargo run --quiet --example ros34pw3_compliance`

The full cargo, clippy, format, and diff checks are run by the parent parity
wave. The Julia fixture was not run in this environment; retry
`julia --project=. tests/julia/ros34pw3.jl` (with the pinned OrdinaryDiffEq
checkout and its dependencies available) to enable that gate.

Status: implementation complete.

Commit: `8d3980c4d028de615194dbbec059c3d4dd6f4249`
