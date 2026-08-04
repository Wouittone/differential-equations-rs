# Shared driver low-storage Runge--Kutta wave handoff

## Summary

- Migrated all nine public fixed-step Williamson 2N low-storage Runge--Kutta methods from the standalone lifecycle in `low_storage_rk.rs` to the frozen crate-private first-order driver.
- Added a fixed-only `LowStorageKernel` that owns only derivative and recurrence-residual scratch; the shared driver owns the candidate state and all time, callback, saving, termination, attempt, statistics, and solution-assembly behavior.
- Preserved the existing recurrence validation and numerical update order. Kernel initialization intentionally performs no RHS evaluation because every attempt evaluates stage zero itself.
- Kept accept and reject hooks as no-ops because this recurrence has no cross-step cache state.

## Files changed

- `src/low_storage_rk.rs`
- `tests/low_storage_rk_allocations.rs`
- `docs/handoffs/driver_low_storage_wave.md`

## Public APIs added

None. The driver adapter and recurrence kernel are crate-private, and the public algorithm set is unchanged.

## Upstream source and revision

- Repository: `https://github.com/SciML/OrdinaryDiffEq.jl`
- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Verified local checkout: `D:/Source/_review/OrdinaryDiffEq.jl`
- Constructor metadata: `lib/OrdinaryDiffEqLowStorageRK/src/algorithms.jl`.
- Williamson 2N recurrence and in-place cache lifecycle: `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl` lines 1-76.
- Pinned coefficients: `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl`, including the 2N cache constructors beginning at lines 27, 127, 365, and 458 and their adjacent family definitions.

## Rust tests

- All 70 library tests passed, including all existing design-order, forward/backward, endpoint clipping, callback, `save_at`, statistics, nonfinite, and fixed-step behavior exercised by the low-storage family and shared driver.
- Added a malformed recurrence dimension test proving validation happens before driver dispatch.
- Added RHS-counter coverage proving a terminating accepted-step callback performs exactly that method's seven stage evaluations and no post-effect work, and an initially terminating callback performs zero RHS evaluations.
- Added a family-local allocation integration test proving one and one thousand callback-free fixed steps allocate equally and remain within the seven-allocation solve ceiling.
- All other integration tests and all example targets passed (`cargo test --all-targets`).

## Julia tests

- Pinned environment check passed for all 13 OrdinaryDiffEq packages at the exact target revision.
- Full Julia compliance passed: 202/202 checks, including 10/10 low-storage Runge--Kutta checks.
- Julia sources and fixtures were not edited.

## Commands run

- `cargo fmt -- --check` — passed.
- `cargo test --all-targets` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `git diff --check` — passed.
- `julia --project=tests/julia tests/julia/pinned_environment.jl --check` — passed.
- `julia --project=tests/julia tests/julia/runtests.jl` — passed, 202/202.
- `cargo run --quiet --release --example low_storage_rk_compliance` — output compared before and after and was byte-identical for all nine methods.

## Numerical differences

None observed. The coefficient literals, stage-time evaluations, residual recurrence, candidate accumulation order, endpoint values, design-order checks, and RHS statistics are unchanged. The release compliance output is byte-identical before and after migration.

## Allocation/performance impact

- No per-step allocation was introduced; allocation count is invariant between one and one thousand callback-free steps.
- Solve-time numerical workspace remains four vectors total: driver-owned current/candidate state plus kernel-owned derivative/residual scratch, matching the four vectors in the former standalone lifecycle.
- Static dispatch and the two-register recurrence are retained. Initialization removes the possibility of an unused pre-attempt RHS evaluation.

## Known limitations

- These methods remain fixed-step-only.
- Upstream stage/step limiters, `williamson_condition`, and threading options remain out of scope for this wave.
- Coefficient schema/code generation and method-specific dense interpolation remain Phase 4 and Phase 6 work.
- No algorithms, coefficients, options, or public APIs were added.

## Follow-up dependencies

- Phase 4 may move the unchanged 2N coefficients into the declarative generated schema without changing the kernel contract.
- Phase 6 may supply method-specific dense segments to the driver; current callback localization and `save_at` continue to use shared linear accepted-step segments.

## Recommended next task

Run the independent Phase 2 final audit across all first-order solver modules, confirming that only intentionally deferred families retain standalone lifecycles and that every migrated kernel obeys immediate-termination and allocation invariants.
