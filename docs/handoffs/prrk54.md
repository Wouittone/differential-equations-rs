# pRRK54 handoff

## Summary

Ported the fixed-step parametric-relaxation SSPRK(5,4) algorithm as `Prrk54`
with the upstream-compatible `pRRK54` alias.  The implementation applies the
full per-step psi/alpha/beta/abscissa transform, supports nonzero `kappa`, and
uses the shared lifecycle for endpoint saving, backward integration, callbacks,
and `save_at` output.

## Files changed

- `src/ssprk_extended.rs`
- `src/lib.rs`
- `tests/ssprk_prrk.rs`
- `tests/ssprk_prrk54_allocations.rs`
- `examples/ssprk_compliance.rs`
- `tests/julia/ssprk.jl`

## Public APIs added

- `Prrk54 { kappa: f64 }`
- `Prrk54::new(kappa)` and `Default` (`kappa = 0`)
- `pRRK54` type alias

## Upstream source and revision

SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl`, `pRRK54ConstantCache`
- `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl`, `_prrk54_coeffs`
  and `perform_step!(..., pRRK54ConstantCache, ...)`
- `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl`, `pRRK54`

## Rust tests

- Fourth-order fixed-step convergence at `kappa = 0`.
- Nonzero relaxation, backward integration, `save_at`, terminating callback,
  and adaptive-mode rejection.
- Fixed-step allocation count is invariant with the number of steps.

## Julia tests

Added `pRRK54` to `tests/julia/ssprk.jl` using the pinned
`OrdinaryDiffEqSSPRK` package and fixed-step reference solve.  The Julia
executable was not available in this worker (`Get-Command julia` returned no
command), so the fixture must be run when Julia is on PATH.

## Commands run

- `cargo fmt --all -- --check` — passed.
- `cargo test --all-targets` — passed (99 unit tests plus all integration and
  example targets).
- `cargo clippy --all-targets -- -D warnings` — passed using an isolated
  target directory because the shared worktree target lock was concurrently
  unavailable.
- `git diff --check` — passed.
- `cargo run --quiet --target-dir ... --example ssprk_compliance` — passed;
  emitted the `prrk54` endpoint row.
- Julia pinned-environment and full compliance gates — not run because Julia
  was absent from PATH.

## Numerical differences

None intended.  The Rust kernel uses the pinned decimal Shu--Osher constants
and the same transformed coefficients and modified abscissae as upstream.
Dense output uses the shared endpoint-compatible recorder path, as upstream
`pRRK54` is fixed-step only and has no dedicated dense interpolant.

## Allocation/performance impact

The kernel allocates all eight dimension-sized work buffers once at solve
initialization and performs no per-step allocation.  The allocation regression
test confirms one-step and 1000-step solves have the same allocation count.

## Known limitations

`pRRK54` is fixed-step only, matching upstream.  Stage and step limiter hooks
from Julia are not exposed by the Rust public API.

## Follow-up dependencies

Coordinator should merge the branch, regenerate the ODE inventory, and run the
Julia pinned and full-suite gates once the Julia executable is available.

## Recommended next task

Continue with another missing regular ODE solver family while the coordinator
performs the inventory/status update.
