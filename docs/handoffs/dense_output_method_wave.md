# Retained method-specific dense-output wave

## Summary

This wave adds the pinned free continuous extensions for `Dp5`,
`OwrenZen3`, `OwrenZen4`, and `OwrenZen5`. Their accepted stage data now drive
one consistent interpolation service for `save_at`, scalar continuous-root
localization, and opt-in retained post-solve queries. No linear or cubic
Hermite fallback is relabeled as a method-specific high-order interpolant.

This document records the state of the initial explicit-method wave. Its
then-remaining gaps were subsequently closed by the shared dense lifecycle,
native FIRK/extrapolation/Taylor retention, typed derivative hooks, and
partition-aware second-order recorders. See
[`shared_dense_lifecycle.md`](shared_dense_lifecycle.md) and
[`roadmap_completion.md`](roadmap_completion.md) for the completed state.

## Files changed

- `src/explicit_rk.rs`: pinned polynomial rows and tableau wiring.
- `tests/method_specific_dense.rs`: Rust interpolation, direction, callback,
  query, and RHS-cost coverage.
- `tests/julia/explicit_dense.jl`: matched pinned Julia samples and roots.
- `docs/FEATURE_COVERAGE.md` and `docs/ALGORITHM_COVERAGE.md`: accurate current
  coverage statements.
- this handoff.

## APIs

No public signature changed. Existing `SolveOptions::retain_dense_output`,
`Solution::interpolate`, `save_at`, and scalar continuous callbacks acquire
method-specific behavior for the four newly wired algorithms.

## Algorithms covered

- New in this wave: `Dp5`, `OwrenZen3`, `OwrenZen4`, `OwrenZen5`.
- Previously covered by the same retained first-order service: `Tsit5`.
- Previously covered only for in-solve second-order sampling/root location:
  `Dprkn6`.

The four new extensions are free: they use stages already evaluated by the
accepted step and perform no lazy RHS calls.

## Upstream sources

Pinned revision: SciML/OrdinaryDiffEq.jl
`211142263781255a9aa2f910f6760b9f18ec29c8`.

- DP5 interpolation basis and stage combinations:
  `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:22-92` and
  `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_addsteps.jl:83-125`.
- Exact DP5 dense `d` coefficients:
  `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl`,
  `DP5_dense_ds`.
- Owren--Zennaro interpolation polynomials:
  `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:286-1328`.
- Exact Owren--Zennaro interpolation rows:
  `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:63-505`.

## Rust and Julia tests

Rust coverage checks interpolation convergence, exact segment endpoints,
forward/backward interval lookup, `save_at`/post-solve consistency, continuous
event localization, callback left/right discontinuity semantics, and unchanged
RHS evaluation counts with dense sampling/retention.

The Julia fixture locks the same one-step samples at `t = 0.2, 0.55, 0.9` and
continuous termination roots for fixed `dt = 0.25` for all four methods.

## Numerical differences

The Rust fixtures use the pinned rational coefficient forms. Expected
one-step exponential samples are:

| Algorithm | `u(0.2)` | `u(0.55)` | `u(0.9)` |
|---|---:|---:|---:|
| DP5 | 1.221470148631057 | 1.7331417529458806 | 2.4595887579652946 |
| OwrenZen3 | 1.216 | 1.706291666666667 | 2.413 |
| OwrenZen4 | 1.22112999591419 | 1.7325755385254473 | 2.460288616366704 |
| OwrenZen5 | 1.2213113186813191 | 1.7330660720366877 | 2.4590016672390163 |

Matched root-time tolerances are `5e-12`; matched dense-state tolerances are
`3e-14`. No existing solver or compliance tolerance was weakened.

## Allocation impact

In-step interpolation borrows the existing stage-major workspace and allocates
no stage vectors. Dense samples and roots add zero RHS evaluations. Opt-in
post-solve retention continues to clone one segment's state/stage storage per
accepted step, as designed; the default non-retained solve path is unchanged.

## Historical residual coverage gaps

1. `Bs5` needs upstream stages 9--11 and its RKSuite polynomial rows. Those
   stages are lazy and are not represented by the current shared tableau-only
   step cache.
2. `Vern6/7/8/9` need their distinct interpolation tables and lazy extra-stage
   materialization. Endpoint stages alone are insufficient for the claimed
   upstream orders.
3. SSPRK methods need an algorithm-to-interpolant dispatch port from
   `OrdinaryDiffEqSSPRK/src/interpolants.jl`; methods without a special
   dispatch must remain on the honest generic fallback.
4. Rosenbrock23/32 and extended Rosenbrock/Rodas methods need owning segment
   forms for their scaled stiff stage combinations and dense coefficient rows.
5. TRBDF2/SDIRK, Adams/QNDF/BDF, stabilized, low-storage, split, and
   symplectic/partitioned kernels do not yet expose accepted endpoint
   derivatives/step history through a retained segment hook. Upstream generic
   Hermite behavior should be ported where that is the actual dispatch; it is
   not method-specific high-order output.
6. `Dprkn6` uses its pinned extension for `save_at` and continuous callbacks,
   but `SecondOrderSolution` has no retained-segment query API. Other RKN and
   second-order families still use their existing endpoint fallback.
7. Callback discontinuity semantics are implemented for retained first-order
   segments (left polynomial, exact right endpoint), but the families above
   cannot inherit them until they use the shared accepted-segment lifecycle.

## Commands

- `cargo test --test method_specific_dense`: 6 passed.
- `cargo test --test explicit_dense`: 7 passed.
- `cargo test --all-targets --all-features`: 334 passed across 112 suites.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- `julia --project=tests/julia tests/julia/explicit_dense.jl`: not runnable;
  neither `julia` nor a WindowsApps Julia shim exists on this host. The
  matched fixture is committed for the configured Julia CI/runtime and this
  execution gap is not represented as a numerical pass.

## Integration notes

The change is isolated on `codex/dense-output-wave`. It only adds immutable
tableau metadata to the existing static-dispatch RK kernel; no solver endpoint,
controller, rejection, or FSAL code changes. The branch can be cherry-picked
as one semantic commit after validation.
