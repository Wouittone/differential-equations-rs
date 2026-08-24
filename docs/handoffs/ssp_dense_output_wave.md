# SSP dense-output family wave

## Summary

Every currently implemented native SSP public algorithm now follows the dense
dispatch selected by pinned `OrdinaryDiffEqSSPRK`. `SspRk22`, `SspRk33`,
`SspRk43`, and `SspRk432` use the package's free quadratic continuous
extension. All remaining implemented SSP algorithms use retained cubic Hermite
segments, matching the upstream core fallback without calling that fallback a
method-specific high-order interpolant.

Both paths are used consistently for `save_at`, scalar continuous roots, and
opt-in `Solution::interpolate` queries. Callback truncation retains the
left-limit polynomial on its original attempted interval and returns the exact
post-effect state at the event time.

## Files changed

- `src/explicit_rk.rs`: specialized SSP coefficient rows and generic explicit
  Hermite root/retention lifecycle.
- `src/ssprk_extended.rs`: SSPRK432 specialization and Hermite support for the
  three parametric-relaxation kernels.
- `src/ssprk_msvs.rs`: free endpoint-Hermite support for both multistep SSP
  kernels.
- `src/solution.rs`: owning Hermite segments and a common owning dense-segment
  enum with callback-truncated bounds.
- `src/integrator.rs`, `src/problem.rs`: counted dense preparation and a
  continuous-callback capability query.
- `tests/ssprk_dense.rs`, `tests/julia/ssprk_dense.jl`: Rust and matched Julia
  coverage.
- feature/algorithm coverage documentation and this handoff.

## APIs

No public signature changed. Existing `SolveOptions::retain_dense_output`,
`Solution::interpolate`, `save_at`, and continuous callbacks gain consistent
SSP dense behavior.

## Algorithm-to-interpolant audit

| Pinned dispatch | Implemented Rust public types | Local segment |
|---|---|---|
| SSP special union | `SspRk22`, `SspRk33`, `SspRk43`, `SspRk432` | Free quadratic stage polynomial |
| Core generic | `SspRk53`, `SspRk53H`, `SspRk53TwoN1`, `SspRk53TwoN2`, `SspRk54`, `SspRk63`, `SspRk73`, `SspRk83`, `SspRk104`, `SspRk932` | Cubic Hermite |
| Core generic | `KykSsprk42`/`KYKSSPRK42`, `Kyk2014DgSsprk3S2` | Cubic Hermite |
| Core generic | `Prrk22`/`pRRK22`, `Prrk33`/`pRRK33`, `Prrk54`/`pRRK54` | Cubic Hermite |
| Core generic | `SspRkMsvs32`/`SSPRKMSVS32`, `SspRkMsvs43`/`SSPRKMSVS43` | Cubic Hermite |

Aliases share their canonical type and do not represent additional numerical
dispatches.

## Coefficient provenance

Pinned upstream revision:
`211142263781255a9aa2f910f6760b9f18ec29c8`.

`lib/OrdinaryDiffEqSSPRK/src/interpolants.jl:1-198` defines one specialized
union containing only SSPRK22/33/43/432. Its value formula is

`u0 * (1 - theta^2) + u1 * theta^2 + dt * k1 * theta * (1 - theta)`.

The Rust rows algebraically expand `u1-u0 = dt * sum(b_i*k_i)` into the existing
stage-polynomial representation: stage one uses `[1, b1-1]` and every other
stage uses `[0, b_i]`. No extra stage or RHS evaluation is introduced.

Types absent from that specialized union fall through
`OrdinaryDiffEqCore/src/dense/generic_dense.jl` to cubic Hermite when endpoint
derivatives are available. The Rust path therefore evaluates one endpoint
derivative per accepted shared-explicit or relaxation step only when dense
sampling, retention, or a continuous callback requires it. MSVS already
evaluates the endpoint derivative as part of stepping and adds none.

## Rust and Julia tests

Rust tests cover specialized interpolation convergence, exact endpoints,
forward/backward evaluation, matched formula samples, `save_at`, continuous
roots, retained queries, callback left/right semantics, all canonical SSP
types, and RHS-count regressions.

Julia tests lock dense samples and roots for all four specialized methods plus
generic-Hermite samples/roots for shared-tableau SSPRK53 and SSPRKMSVS32.
`pRRK22` samples are also matched. At the pinned upstream revision, a pRRK22
continuous callback raises `FieldError: pRRK22Cache has no field tmp`; the Rust
root behavior is tested, but no false Julia root-pass is claimed.

## Commands

- `cargo test --test ssprk_dense`: 7 passed.
- `cargo test --all-targets --all-features`: 351 passed across 116 suites.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- `julia.exe --project=tests/julia tests/julia/ssprk_dense.jl`: specialized
  12/12 and generic 5/5 passed.
- `julia.exe --project=tests/julia tests/julia/runtests.jl`: the complete pinned
  Julia comparison suite passed, including all new SSP cases and all existing
  solver-family fixtures.

## Numerical differences

The specialized one-step exponential samples and roots match Julia at
`3e-14` and `2e-13` absolute tolerances respectively. Generic quadratic
Hermite samples and roots match at `1e-14` and `2e-14`. No existing tolerance
was weakened.

## Allocation and RHS impact

- SSPRK22/33/43/432: no extra RHS evaluations for samples, roots, or retained
  segments; evaluation borrows accepted stages.
- Shared explicit generic methods and pRRK methods: one counted endpoint RHS
  evaluation per accepted step that actually needs dense service, reused by
  roots, saving, and retention rather than duplicated.
- MSVS32/43: no extra RHS evaluations because their step already owns both
  endpoint derivatives.
- Retention allocates owning state/derivative data per accepted step only when
  explicitly enabled. Default non-dense solves retain no segments.

## Exact remaining SSP gaps

There is no remaining dense-dispatch gap for the currently implemented native
regular-ODE SSP public types. The upstream pRRK22 mutable-cache callback failure
prevents a matched Julia root execution at the pinned revision, but is not a
missing Rust interpolation path. Stage/step limiter callbacks, arbitrary state
containers, and upstream threading options remain broader SSP API gaps rather
than dense-output gaps.

## Integration notes

This wave is isolated on `codex/ssp-dense-output`. The owning dense enum is an
internal representation change; public `Solution` layout and query contracts
remain stable.
