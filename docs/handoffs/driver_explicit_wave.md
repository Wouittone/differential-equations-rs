# Shared driver and generic explicit-RK handoff

## Summary

Added a crate-private, statically dispatched first-order integration driver and
migrated the generic explicit Runge-Kutta implementation (fixed and embedded
adaptive tableaus) onto it. The driver now owns both state buffers, direction
and endpoint progression, attempted-step limits, rejection/acceptance stats,
callback/event ordering, saving, termination, controller proposals, and final
solution assembly. Explicit RK retains stage storage, initial-step estimation,
error estimators, FSAL state, and cache invalidation in its kernel.

The driver lifecycle is covered by mock-kernel tests before the solver-family
tests. A terminating effect returns after saving the affected state and before
the kernel accept hook. Rejected attempts do not invoke callbacks or record
trajectory data.

## Files changed

- `src/integrator.rs` (new): internal kernel contract, common driver, mock tests.
- `src/explicit_rk.rs`: generic explicit RK kernel and driver migration.
- `src/lib.rs`: private `integrator` module declaration only.
- `tests/integrator_driver.rs` (new): solve-level allocation regression test.
- `docs/handoffs/driver_explicit_wave.md` (new): this handoff.

## Public APIs added

None. `StepKernel`, `KernelCapabilities`, `StepEstimate`, and the driver are all
crate-private. Existing named algorithms and custom `ButcherTableau` use are
unchanged.

## Upstream source and revision

Behavior remains targeted at SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`. This architecture wave changes no
coefficients or public algorithms.

## Rust tests

- Mock kernel: adaptive rejection isolation, backward endpoint clipping,
  callback pre-effect interpolation/post-effect force-save, continuous-event
  termination ordering, initial termination ordering, `save_at`, step-size
  underflow, maximum attempted steps, and two-buffer reuse.
- Allocation regression: callback-free fixed RK4 uses the same six allocations
  for one step and one thousand steps (the same count and bytes as the pre-wave
  implementation).
- Existing generic explicit-RK tests cover fixed/adaptive named methods, dual
  estimators, FSAL methods, malformed and public custom tableaus, non-finite
  derivatives, and convergence.
- Existing cross-family callback/saving integration suite passed, including all
  migrated explicit-RK cases.

## Julia tests

- Pinned environment check passed for all 13 OrdinaryDiffEq packages at the
  requested revision.
- Full Julia compliance passed: 202 checks across all 15 suites, including 23
  low-order explicit-RK checks, 17 Owren-Zennaro/BS5 checks, 4 SSP checks, and
  3 callback/save-at checks.

## Commands run

All required gates passed:

```text
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

Additional checks:

```text
cargo test --lib
cargo test --test integrator_driver -- --nocapture
cargo run --quiet --example low_order_compliance
cargo run --quiet --example benchmark_matrix -- 1
```

## Numerical differences

None observed. Every scalar emitted by `low_order_compliance` was bit-for-bit
identical to commit `408991c`; endpoint checksums, RHS evaluation counts, and
accepted/rejected behavior in the benchmark and test matrices were unchanged.

## Allocation/performance impact

No per-step heap allocation was introduced. Callback-free fixed RK4 retains
the pre-wave total of six solve-time allocations independent of step count.
The benchmark matrix also retained the exact pre-wave allocation count and
allocated bytes for fixed and adaptive generic explicit methods. The extra
initial-step derivative temporarily reuses the driver-owned candidate buffer,
so adaptive initialization does not require another vector.

## Known limitations

- Saving and continuous-event localization still use the existing linear state
  segment; method-specific dense interpolation belongs to Phase 6.
- Controller constants and rejection limiting deliberately preserve the old
  explicit-RK policy; PI/PID and time-stop parity are not added here.
- The driver currently exposes only the capability data needed by existing
  fixed/embedded one-step methods. Later families may extend the crate-private
  capability record without changing public API.
- Only generic explicit RK is migrated. Tsit5, Verner, low-storage RK, extended
  SSP, implicit, Rosenbrock/Rodas, TRBDF2, Adams, variable Adams, and
  second-order methods remain on their existing loops.

## Follow-up dependencies

Later first-order kernel migrations should implement the frozen lifecycle and
must preserve the immediate-return termination rule. Dense-output work should
replace the driver's linear segment at the callback/recorder boundary rather
than move callback ownership back into family kernels.

## Recommended next task

Migrate one-step implicit methods to `StepKernel` after the Phase 3 internal
matrix/linear-solver interfaces are frozen, followed by Rosenbrock/Rodas and
then fixed/variable Adams kernels.
