# Rosenbrock shared-driver migration handoff

## Summary

Migrated `Rosenbrock23`, `Rosenbrock32`, `Rodas4`, and `Rodas5P` from duplicated
integration lifecycle loops to the frozen shared first-order driver. The family
modules now contain only solver facades, `StepKernel` implementations, numerical
workspaces, stage equations, differentiation/factorization logic, estimators,
and focused tests.

The kernels preserve the proportional-controller policy (`safety = 0.9`,
factors `0.2..6.0`, error orders 3/3/4/5), terminal singular-factorization
behavior, differentiation reuse across error rejection, and cache invalidation
after acceptance. `Rosenbrock23` retains its adaptive callback-free endpoint
derivative swap; callback effects force a fresh RHS evaluation. Terminating
callbacks return before the accept hook and all post-effect RHS/Jacobian/
factorization work.

## Files changed

- `src/rosenbrock.rs`
- `src/rosenbrock_extended.rs`
- `tests/rosenbrock_driver.rs`
- `docs/handoffs/driver_rosenbrock_wave.md`

## Public APIs added

None. The existing public algorithm types and solve behavior are unchanged.

## Upstream source and revision

- SciML/OrdinaryDiffEq.jl revision
  `211142263781255a9aa2f910f6760b9f18ec29c8`.
- Constructors audited from
  `lib/OrdinaryDiffEqRosenbrock/src/algorithms.jl` (`Rosenbrock23` line 6,
  `Rosenbrock32` line 12, `Rodas4` line 48, `Rodas5P` line 84 in the pinned
  inventory).
- Existing coefficients and stage equations were deliberately unchanged; this
  wave only extracted lifecycle ownership.

## Rust tests

- Existing stiff nonautonomous, fixed-order, backward-integration,
  analytic/numerical Jacobian, callbacks, `save_at`, and rejection-reuse tests
  pass.
- Added termination work-counter tests for `Rosenbrock23` and `Rodas4`; a
  terminating effect whose state would make the next RHS non-finite succeeds
  with one accepted step and one Jacobian evaluation.
- Added a Rodas rejected-step differentiation-reuse regression.
- Added a callback-effect Jacobian-observation regression proving the next
  factorization is built from the affected state.
- Added `tests/rosenbrock_driver.rs`: one versus 1,000 callback-free fixed steps
  have identical allocation counts for all four algorithms (and remain at or
  below 25 solve-time allocations).
- `cargo test --all-targets`: pass (72 library tests, all integration tests,
  allocation tests, and example targets).

## Julia tests

- Pinned environment check: pass; 13 packages verified at the target revision.
- Full Julia suite: pass, including Rosenbrock23 3/3 and extended Rosenbrock
  13/13 compliance assertions.
- No Julia fixtures were edited.

## Commands run

- `cargo fmt -- --check` — pass.
- `cargo test --all-targets` — pass.
- `cargo clippy --all-targets -- -D warnings` — pass.
- `git diff --check` — pass.
- `julia --project=tests/julia tests/julia/pinned_environment.jl --check` — pass.
- `julia --project=tests/julia tests/julia/runtests.jl` — pass.
- `cargo run --quiet --example rosenbrock_compliance` before/after — identical.
- `cargo run --quiet --example rosenbrock_extended_compliance` before/after —
  identical.

## Numerical differences

None observed. The compliance executables were byte-for-byte identical before
and after:

```text
rosenbrock23,5.40302305918423098e-1,76767
rosenbrock32_adaptive,5.40301246631114007e-1
rosenbrock32_fixed,2.71828183173328775e0
rodas4_adaptive,5.40302305990256637e-1
rodas4_fixed,2.71828182843933241e0
rodas5p_adaptive,5.40302305899419633e-1
rodas5p_fixed,2.71828182845906818e0
```

## Allocation/performance impact

No per-step allocation was introduced. The driver-owned candidate buffer
replaces the former workspace-owned candidate buffer, keeping one candidate
allocation per solve. All four family algorithms have allocation counts that
are invariant between one and 1,000 callback-free fixed steps.

## Known limitations

- Method-specific dense interpolation remains outside this architecture wave.
- Linear algebra and factorization strategy were not redesigned.
- Singular factorization remains terminal, matching the pre-wave modules.

## Follow-up dependencies

The Rosenbrock/Rodas families now depend only on the frozen shared first-order
driver contract and are ready for the later dense-output/controller and linear
interface phases.

## Recommended next task

Migrate the fixed and variable Adams families to the shared driver, then run
the coordinator-owned complete architecture-wave verification and status
update.
