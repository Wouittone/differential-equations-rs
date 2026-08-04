# Shared driver implicit-family wave handoff

## Summary

- Extended the crate-private first-order driver with per-family proportional-controller metadata and an explicit recoverable-attempt failure policy.
- Adaptive kernels may classify `NonlinearSolveFailed` and `SingularLinearSystem` as recoverable. The driver counts the attempt as rejected, calls the kernel rejection hook, shrinks by the family failure factor, and retries without checking the failed candidate, running callbacks, or saving. The same errors remain terminal in fixed-step mode.
- Migrated `ImplicitEuler`, `ImplicitMidpoint`, `Trapezoid`, and `Trbdf2` to `StepKernel`. Their modules no longer duplicate time progression, callback processing, saving, attempt accounting, termination, or solution assembly.
- Kept Newton iterations, numerical/analytic Jacobians, LU state, error estimation, derivative reuse, and cache invalidation inside the family kernels.
- The driver interface is frozen for the remaining Phase 2 Rosenbrock/Rodas, fixed/variable Adams, and specialized fixed-explicit migrations. `ControllerConfig` carries error order, safety, minimum/maximum factors, post-rejection acceptance/rejection caps, and failed-attempt shrink. Existing `KernelCapabilities::new` remains exactly equivalent to the explicit-RK policy.

## Files changed

- `src/integrator.rs`
- `src/implicit.rs`
- `src/trbdf2.rs`
- `tests/integrator_driver.rs`
- `docs/handoffs/driver_implicit_wave.md`

## Public APIs added

None. All driver, kernel, controller, and failure-policy changes are crate-private.

## Upstream source and revision

- Repository: `https://github.com/SciML/OrdinaryDiffEq.jl`
- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Verified local checkout: `D:/Source/_review/OrdinaryDiffEq.jl`
- Constructor references: `lib/OrdinaryDiffEqSDIRK/src/algorithms.jl` lines 63, 108, 150, and 203.
- TRBDF2 coefficients and predictor references: `lib/OrdinaryDiffEqSDIRK/src/sdirk_tableaus.jl` lines 1-44.
- Smoothed-estimate behavior: `TRBDF2(smooth_est=true)` in `lib/OrdinaryDiffEqSDIRK/src/algorithms.jl` and the ESDIRK smoothed-estimate path in `lib/OrdinaryDiffEqSDIRK/src/generic_imex_perform_step.jl`.

## Rust tests

- All 68 library tests passed, including existing implicit/TRBDF2 stiffness, order, Jacobian, backward, callback, and save-at coverage.
- All 10 callback/saving integration tests and all 6 second-order integration tests passed.
- Added driver mock coverage for nonlinear and singular recoverable failures, stale failed-candidate isolation, callback/save suppression on failure, failure shrink followed by underflow, and terminal fixed-step failures.
- Added family counter tests proving terminating callback effects are followed by no RHS or Jacobian work for fixed implicit and TRBDF2 kernels.
- Extended the allocation regression to prove callback-free fixed implicit and adaptive TRBDF2 solve allocations remain constant as accepted-step counts grow.

## Julia tests

- Pinned environment check passed for all 13 OrdinaryDiffEq packages at the exact target revision.
- Full Julia compliance passed: 202/202 checks, including 4 fixed implicit and 5 TRBDF2 checks.
- Julia sources and tests were not edited.

## Commands run

- `cargo fmt -- --check` — passed.
- `cargo test --all-targets` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `git diff --check` — passed.
- `julia --project=tests/julia tests/julia/pinned_environment.jl --check` — passed.
- `julia --project=tests/julia tests/julia/runtests.jl` — passed, 202/202.
- `cargo run --quiet --example implicit_compliance` and `cargo run --quiet --example trbdf2_compliance` — outputs matched the pre-migration `47c8ae5` worktree byte-for-byte, including TRBDF2 accepted/rejected counts (`21/3`).

## Numerical differences

None observed. The implicit formulas, Newton tolerance and iteration cap, Jacobian construction, factorization reuse, TRBDF2 tableau, smoothed estimator, initial-step estimate, and controller arithmetic are unchanged. Explicit RK retains safety `0.9`, factor bounds `0.2..10.0`, and its tableau order exponent. TRBDF2 retains safety `0.9`, factor bounds `0.2..6.0`, error exponent `1/3`, post-rejection growth cap `1.0`, and failed-attempt factor `0.2`.

## Allocation/performance impact

- No per-step allocations were introduced.
- Both migrated workspaces no longer own a duplicate candidate vector; the driver-owned candidate is used directly, removing one solve-time vector allocation per migrated solve.
- Fixed implicit and adaptive TRBDF2 allocation counts are invariant between short and high-step-count callback-free solves and remain within the regression ceilings (20 and 25 allocations respectively).
- Static dispatch and existing dense LU/Newton hot paths are retained.

## Known limitations

- Fixed one-step implicit methods remain fixed-step-only in this wave, as scoped by the task.
- The proportional controller metadata preserves current Rust behavior; PI/PID history, `dtmin`, time stops, and method-specific dense output remain Phase 6 work.
- Dense matrix and solver abstractions remain in `linear.rs` for Phase 3; this wave deliberately did not redesign them.
- No Rosenbrock/Rodas, Adams, low-storage/SSP, or other solver module was migrated here.

## Follow-up dependencies

- Rosenbrock/Rodas kernels should select a `ControllerConfig` with their existing `0.9`, `0.2..6.0`, and method-specific error order, retaining terminal singular-factorization behavior unless their current module explicitly retries it.
- Variable Adams kernels should select their existing `0.9`, `0.2..5.0`, and configured method order. Fixed Adams and specialized fixed-explicit kernels can use `KernelCapabilities::new(false, order)` because controller fields are inactive in fixed mode.
- Kernels with adaptive Newton stages may opt into `recover_nonlinear_and_singular_failures`; kernels must leave the default terminal policy when recovery is not current behavior.

## Recommended next task

Migrate Rosenbrock23 and extended Rosenbrock/Rodas methods onto the frozen driver contract, preserving their family `maximum_factor = 6.0`, Jacobian/factorization reuse, failure semantics, counters, callback termination, and allocation behavior.
