# SSPRKMSVS32 handoff

## Summary

Added the native regular-ODE `SspRkMsvs32` implementation (and the
`SSPRKMSVS32` spelling alias).  The kernel follows the pinned
OrdinaryDiffEqSSPRK three-step, second-order startup and constant-step
`Omega = 2` recurrence, retaining endpoint derivatives and resetting
multistep history after callbacks.

## Files changed

- `src/ssprk_msvs.rs`
- `src/lib.rs`
- `examples/ssprk_extended_compliance.rs`
- `tests/julia/ssprk_extended.jl`

## Public APIs added

- `SspRkMsvs32`
- `SSPRKMSVS32` type alias

## Upstream source and revision

- `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:118-133`
- `lib/OrdinaryDiffEqSSPRK/src/alg_utils.jl:16-18,49`
- `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:885-950`
- `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:948-1076`
- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`

## Rust tests

The module tests cover second-order convergence, backward integration, and
the fixed-only/adaptive configuration boundary.  The compliance example
covers a nonautonomous endpoint.

## Julia tests

`tests/julia/ssprk_extended.jl` adds `SSPRKMSVS32` to the pinned reference
fixture.  Julia is unavailable in this environment (`julia` is not
recognized); retry the pinned and full Julia commands after restoring Julia.

## Commands run

- `cargo fmt --all`
- `cargo test ssprk_msvs::tests -- --nocapture`
- Full `cargo test --all-targets` was run before the final test-only adjustment;
  rerun after integration.

## Numerical differences

The pinned type is marked as adaptive-capable in Julia but its documentation
says it has no error estimator and requires a fixed timestep.  This port
follows the actual fixed-step implementation and returns
`AdaptiveStepUnsupported` for adaptive options.  A tiny clipped final step
uses the two-stage startup update rather than applying the multistep history
across a roundoff-sized interval.

## Allocation/performance impact

The kernel allocates five reusable state/derivative buffers per solve and no
per-step vectors.

## Known limitations

Stage/step limiter and threading hooks from the Julia wrapper are outside the
regular scalar/vector ODE API.  Julia compliance remains pending the missing
Julia executable.

## Follow-up dependencies

Coordinator should regenerate the exact-revision inventory and update
`docs/OVERNIGHT_STATUS.md` after merging the public constructor.

## Recommended next task

Port the independent `SSPRKMSVS43` four-step, third-order method using the same
fixed-step history-kernel pattern.
