# ROS3PRL2 regular-ODE parity handoff

## Summary

Added the native adaptive `Ros3Prl2` four-stage, third-order stiffly accurate
Rosenbrock method. It uses the shared Rosenbrock/Rodas driver, dense-LU
Jacobian path, adaptive embedded estimator, fixed-step mode, backward
integration, callbacks, requested samples, and the existing allocation-free
workspace.

## Files changed

- `src/rosenbrock_extended.rs`: public marker, pinned tableau, algorithm
  implementation, fixed-step order and lifecycle tests.
- `src/lib.rs`: public `Ros3Prl2` export.
- `examples/rosenbrock_extended_compliance.rs`: Rust compliance endpoint.
- `tests/julia/rosenbrock_extended.jl`: `ROS3PRL2()` reference endpoint.

## Public API added

- `differential_equations::algorithms::rosenbrock::Ros3Prl2`

## Upstream source and revision

- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Tableau: `lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl:896-929`,
  `ROS3PRL2RodasTableau(T, T2)`.
- Algorithm metadata: `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl:256-266`.
- Regular-ODE compliance coverage: `lib/OrdinaryDiffEqRosenbrockTableaus/test/ode_rosenbrock_tests.jl:519-536`.

## Rust tests

- `ros3prl2_covers_regular_ode_lifecycle`: analytic Jacobian, fixed stepping,
  `save_at`, and backward integration.
- `methods_have_their_expected_fixed_step_orders`: ROS3PRL2 convergence order.
- Existing extended Rosenbrock aggregate tests include the adaptive stiff
  nonautonomous path and shared callback/cache behavior.

## Julia tests

The aggregate fixture includes both fixed and adaptive `ROS3PRL2()` endpoints.
The local environment has no `julia` executable (`Get-Command julia` reports no
command), so the exact retry commands are:

```powershell
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

## Commands run

- `cargo fmt --all`
- `cargo fmt -- --check`
- `cargo test --lib rosenbrock_extended::tests::`
- `cargo run --quiet --example rosenbrock_extended_compliance`
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`

## Numerical differences

No method-specific coefficient differences were introduced. The shared Rust
controller may select different adaptive step boundaries than Julia, so the
fixture uses the existing adaptive tolerance rather than bitwise equality.

## Allocation/performance impact

No new per-step allocations were added; ROS3PRL2 uses the existing fixed-size
workspace and dense factorization buffers.

## Known limitations

Method-specific stiff dense interpolation remains outside this slice; regular
ODE trajectory recording and `save_at` are supplied by the shared recorder.
Julia compliance is pending the missing local Julia executable.

## Follow-up dependencies

Coordinator should merge the commit, regenerate the pinned inventory for the
new public constructor, and add the method to the overnight status counts.

## Recommended next task

Continue with another independent Rosenbrock/Rodas tableau, such as `Rodas4P2`
or `ROS3PRL`-adjacent methods, while preserving the shared-driver interfaces.
