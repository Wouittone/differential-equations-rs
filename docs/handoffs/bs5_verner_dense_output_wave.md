# BS5 and Verner dense-output wave

## Summary

`Bs5`, `Vern6`, `Vern7`, `Vern8`, and `Vern9` now use their distinct pinned
OrdinaryDiffEq continuous extensions. Their additional interpolation stages are
evaluated only when `save_at`, a scalar continuous callback, or opt-in retained
dense output requests the accepted step's polynomial. The same owning segment
then drives post-solve `Solution::interpolate`, including backward integration
and callback-truncated left-limit behavior.

## Files changed

- `src/dense_coefficients.rs`: exact lazy-stage and interpolation coefficients.
- `src/explicit_rk.rs`: sparse lazy-stage infrastructure and BS5 dispatch.
- `src/verner.rs`: Vern6/7/8/9 dispatch.
- `src/lib.rs`: private coefficient module registration.
- `tests/bs5_verner_dense.rs`: Rust numerical and lifecycle coverage.
- `tests/julia/bs5_verner_dense.jl`, `tests/julia/runtests.jl`: pinned Julia
  reference fixture.
- feature and algorithm coverage documentation plus this handoff.

## APIs and algorithms

No public solver signature changed. Existing `SolveOptions::retain_dense_output`,
`Solution::interpolate`, `save_at`, and scalar continuous callbacks gain the
method-specific BS5 and Verner paths.

| Algorithm | Base stages | Lazy dense stages | Dense polynomial order |
|---|---:|---:|---:|
| `Bs5` | 8 | 3 (upstream k9-k11) | 5 |
| `Vern6` | 9 | 3 (k10-k12) | 6 |
| `Vern7` | 10 | 6 (k11-k16) | 7 |
| `Vern8` | 13 | 8 (k14-k21) | 8 |
| `Vern9` | 16 | 10 (k17-k26) | 9 |

No generic Hermite fallback remains for these five algorithms.

## Upstream sources and coefficient provenance

Pinned revision: `211142263781255a9aa2f910f6760b9f18ec29c8`.

- BS5 stages and RKSuite rows: `OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl`,
  `low_order_rk_addsteps.jl`, and `interpolants.jl`.
- Verner family stages and rows: `OrdinaryDiffEqVerner/src/verner_tableaus.jl`,
  `verner_addsteps.jl`, and `interpolants.jl`.

The checked-in binary64 literals were extracted from the pinned package's
compiled `Float64` tableau structs. Sparse stage references preserve the exact
upstream dependency graph; interpolation rows preserve each stage's distinct
theta polynomial.

## Tests and numerical differences

Rust tests cover dense-order convergence, exact endpoint agreement, forward and
backward retained queries, `save_at`/query identity, pinned Julia samples,
continuous root times, callback discontinuity semantics, and exact lazy-stage
RHS counts. Existing Verner allocation tests also remain green.

The five one-step exponential sample triplets agree with pinned Julia within
`3e-10` absolute error. The largest difference (`1.95e-10`, Vern8 at theta
0.9) comes from cancellation in interpolation rows whose binary64 magnitudes
reach roughly one million. Continuous root times agree within `7e-12`; no
existing tolerance was weakened.

## Commands

- `cargo test --test bs5_verner_dense`: 6 passed.
- focused dense plus Verner allocation suites: 10 passed across 5 suites.
- `cargo test --all-targets --all-features`: 379 passed across 123 suites.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- pinned `julia.exe --project=tests/julia tests/julia/bs5_verner_dense.jl`:
  15/15 passed.

## RHS and allocation impact

Default endpoint-only solves execute no dense stages and retain no segments.
When dense service is requested, each accepted step executes exactly 3, 3, 6,
8, or 10 additional RHS evaluations for BS5 through Vern9 respectively; one
stage set is reused across saving, roots, and retention. Stage storage remains
one flat workspace allocation, enlarged for the method's maximum stage count.
Opt-in retention owns one segment's states and stages per accepted step.

## Exact remaining gaps and integration notes

There is no remaining method-specific dense-output gap for BS5 or Vern6/7/8/9.
Derivative queries of the dense polynomial are not part of the current public
`Solution` API. Broader upstream features such as stage limiters, arbitrary
state/scalar containers, and threading remain outside this wave.

This change is isolated on `codex/bs5-verner-dense` and starts from integrated
main revision `39c7cf13`.
