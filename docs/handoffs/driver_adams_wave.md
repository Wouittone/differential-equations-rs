# Shared driver Adams-family wave handoff

## Summary

- Migrated `Ab3`, `Ab4`, `Ab5`, `Abm32`, `Abm43`, and `Abm54` to the frozen crate-private first-order driver. `src/adams.rs` now contains an `AdamsKernel` and no integration lifecycle loop.
- Migrated `Vcab3`, `Vcab4`, `Vcab5`, `Vcabm3`, `Vcabm4`, and `Vcabm5` to the same driver. `src/variable_adams.rs` now contains a `VariableAdamsKernel` and no integration lifecycle loop.
- Kept fixed-method bootstrap selection, the ABM repeating-bootstrap predictors, derivative history, predictor/corrector formulas, variable-step divided differences, unequal-step coefficients, startup estimators, and initial-step estimation inside family kernels.
- Kept accepted multistep state separate from trial state. Rejections do not advance the step number, accepted step history, derivative, or accepted divided differences.
- Moved callback history invalidation and post-effect derivative initialization into `accept_step`. The driver returns before the accept hook for terminating callbacks, so no post-termination derivative or history work runs.
- Preserved the variable-Adams proportional controller metadata: safety `0.9`, factor bounds `0.2..5.0`, method-order error exponent, post-rejection caps of `1.0`, and the existing initial-step estimator.

## Files changed

- `src/adams.rs`
- `src/variable_adams.rs`
- `tests/adams_driver.rs`
- `docs/handoffs/driver_adams_wave.md`

## Public APIs added

None. The kernels and shared-driver interfaces remain crate-private.

## Upstream source and revision

- Repository: `https://github.com/SciML/OrdinaryDiffEq.jl`
- Revision: `211142263781255a9aa2f910f6760b9f18ec29c8`
- Verified local checkout: `D:/Source/_review/OrdinaryDiffEq.jl`
- Constructors: `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/algorithms.jl` lines 23-171.
- Fixed-step caches and bootstrap/history state: `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_caches.jl` lines 7-339.
- Variable-coefficient caches: the same cache file, lines 341-1023.
- Fixed and variable perform-step formulas: `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl` lines 1-1445.

## Rust tests

- All 71 library tests passed.
- All callback/saving, shared-driver allocation, and second-order integration tests passed.
- Added direct rejection isolation coverage proving that a rejected VCAB5 startup attempt leaves accepted step history, divided differences, derivative, and step number unchanged, and that the retry matches a fresh kernel bit-for-bit.
- Added fixed- and variable-Adams termination counters. The variable test terminates after reaching the multistep path and proves no RHS evaluation occurs after the terminating effect.
- Added `tests/adams_driver.rs`, which proves callback-free fixed AB5 and adaptive VCAB5 solve allocation counts are invariant between short and high-step-count runs.
- All example targets compiled and their unit-test harnesses passed.

## Julia tests

- The pinned environment check passed for all 13 OrdinaryDiffEq packages at the exact target revision.
- Full Julia compliance passed 202/202 checks, including fixed Adams 13/13 and variable-coefficient Adams 73/73.
- Julia sources, fixtures, and manifests were not edited.

## Commands run

- `cargo fmt -- --check` — passed.
- `cargo test --all-targets` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `git diff --check` — passed.
- `julia --project=tests/julia tests/julia/pinned_environment.jl --check` — passed.
- `julia --project=tests/julia tests/julia/runtests.jl` — passed, 202/202.
- `cargo run --quiet --example adams_compliance` — output matched the pre-migration baseline byte-for-byte.
- `cargo run --quiet --example variable_adams_compliance` — endpoints and accepted/rejected counts matched the pre-migration baseline byte-for-byte for all scalar and vector cases.

## Numerical differences

None observed. Fixed-Adams compliance endpoints are bitwise identical. Variable-Adams scalar/vector endpoints and accepted/rejected counts are bitwise identical. Bootstrap formulas, predictor/corrector estimators, unequal-step coefficient construction, and controller arithmetic are unchanged.

RHS statistics remain unchanged on ordinary accepted/rejected paths and continuing callbacks. Terminating callbacks intentionally skip the family accept hook and any derivative/history work that formerly followed the callback, as required by the frozen driver lifecycle.

## Allocation/performance impact

- No per-step allocation was introduced.
- Both families now write directly into the driver-owned candidate buffer, removing their duplicate candidate vectors.
- Fixed Adams preallocates all derivative-history buffers during kernel initialization and rotates them on acceptance; bootstrap steps no longer grow the history with fresh vector allocations.
- Allocation counts for callback-free fixed AB5 and adaptive VCAB5 solves are invariant as accepted-step counts grow.
- Static dispatch and all existing numerical hot paths are retained.

## Known limitations

- Fixed Adams methods remain fixed-step-only.
- This wave preserves endpoint stepping and the existing linear `save_at`/event interpolation supplied by the shared trajectory recorder. Method-specific multistep dense output remains Phase 6 work.
- PI/PID controller history, time stops, and additional variable-order methods remain outside this wave.
- No BDF, QNDF, DAE residual behavior, or public API was added.

## Follow-up dependencies

- Phase 6 multistep dense segments can use the kernel-owned derivative/divided-difference histories after defining a driver-facing dense-segment interface.
- Future BDF/QNDF work should reuse the accepted-versus-trial history separation demonstrated here but requires its own coefficient/order-control design.
- Public algorithm names did not change, so inventory regeneration is not required for this merge.

## Recommended next task

Migrate the remaining specialized fixed explicit/SSP/low-storage solver modules to the frozen first-order driver, then close Phase 2 with an independent lifecycle and allocation audit before beginning dense-output/controller expansion.
