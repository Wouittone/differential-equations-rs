# ROS34PRw regular ODE handoff

## Summary

Added the native `Ros34Prw` Rust solver for upstream `ROS34PRw`, a
four-stage, third-order Rosenbrock-Wanner method with a second-order embedded
estimator. It uses the existing shared Rosenbrock driver, finite-difference or
analytic Jacobians, dense LU factorization, adaptive proportional control, and
the shared callback/save-at lifecycle.

## Files changed

- `src/rosenbrock_extended.rs`: public type, pinned tableau, registration,
  stiff/nonautonomous, fixed-order, backward, callback/save-at, and analytic
  Jacobian tests.
- `src/lib.rs`: `Ros34Prw` export.
- `tests/rosenbrock_driver.rs`: callback-free allocation-invariance coverage.
- `examples/rosenbrock_extended_compliance.rs`: Rust compliance endpoint row.
- `tests/julia/rosenbrock_extended.jl`: Julia `ROS34PRw()` reference row.

## Public API

`differential_equations::Ros34Prw`.

## Upstream source and revision

`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`,
`ROS34PRwRodasTableau`, at
`211142263781255a9aa2f910f6760b9f18ec29c8`.

## Rust tests

- Adaptive stiff nonautonomous solve reaches the cosine equilibrium.
- Fixed-step convergence is third order.
- Forward/backward integration and callback/save-at behavior are covered.
- Analytic Jacobian and finite-difference Jacobian paths agree, with fewer RHS
  calls for the analytic path.
- Callback-free fixed-step allocations are invariant with step count.

## Julia tests

The existing extended Rosenbrock fixture now includes `ROS34PRw()` and compares
adaptive and fixed endpoint values against the Rust compliance example. The
coordinator should run the pinned and full Julia suites when Julia is available.

## Commands run

- `cargo fmt --all`
- `cargo test --all-targets`
- `cargo test rosenbrock_extended::tests --lib`

## Numerical differences

No method-specific numerical differences were introduced. As with the other
extended Rosenbrock ports, method-specific stiff dense interpolation is not yet
represented; shared trajectory recording is used for endpoint and `save_at`
samples.

## Allocation/performance impact

The tableau is static and adds no solve-path allocations. The existing
four-stage workspace is reused across attempts and steps.

## Known limitations

The upstream regular-ODE algorithm's optional advanced AD/operator and DAE
paths are outside this port. The public Rust spelling is `Ros34Prw` while the
upstream Julia constructor is `ROS34PRw`.

## Follow-up dependencies

The coordinator must regenerate the inventory after merging this public
algorithm and include the Julia fixture in the aggregate suite if required by
the current test runner.

## Recommended next task

Port another independent Rosenbrock/Rodas tableau, then perform a shared dense
interpolation/controller audit for the completed family.
