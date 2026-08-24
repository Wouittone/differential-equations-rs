# Rosenbrock and Rodas dense-output wave

## Summary

Every native Rosenbrock/Rosenbrock-W/Rodas algorithm now follows its pinned
OrdinaryDiffEqRosenbrock value-interpolation dispatch. Rosenbrock23/32 use the
Shampine quadratic extension; tableaus with nonempty `H` matrices retain their
exact precombined stiff correction rows; tableaus with empty `H` matrices use
the upstream cubic-Hermite fallback. The accepted segment is shared by
`save_at`, scalar continuous roots, and opt-in post-solve
`Solution::interpolate`, including backward integration and callback bounds.

## Files and APIs

- `src/solution.rs`: borrowed/owning stiff correction segments.
- `src/rosenbrock_dense.rs`: pinned `H` rows and Rosenbrock23/32 coefficients.
- `src/rosenbrock.rs`: Rosenbrock23 dense lifecycle hooks.
- `src/rosenbrock_extended.rs`: dispatch audit and hooks for all extended types.
- `tests/rosenbrock_dense.rs`: numerical, lifecycle, coverage, and RHS tests.
- `tests/julia/rosenbrock_dense.jl`: pinned Julia samples and root times.
- `docs/FEATURE_COVERAGE.md`: family coverage status.

No public signature changed. Existing `SolveOptions::retain_dense_output`,
`Solution::interpolate`, `save_at`, and continuous callback APIs gain the stiff
dense paths.

## Algorithm dispatch

| Dispatch | Algorithms |
|---|---|
| Shampine quadratic | Rosenbrock23, Rosenbrock32 |
| 2-row `H` extension | Rodas4, Rodas42, Rodas4P, Rodas4P2, Rodas4PW |
| 3-row `H` extension | Rodas5, Rodas5P, Rodas5Pe, Rodas5Pr, Rodas23W, Rodas3P, Tsit5DA |
| 4-row `H` extension | Rodas6P |
| Generic cubic Hermite | Ros2, Rodas3/3d, Ros3/3P/3Pr/3Prl/3Prl2, Ros34Prw/Pw1a/Pw1b/Pw2/Pw3, Grk4a/4t, Rok4a, RosenbrockW6S4OS, Ros2Pr/2S, Ros4LStab, RosShamp4, Scholz4_7, Veldd4, Velds4 |

The Hermite group is not described as method-specific high order: its pinned
tableau has an empty `H`, and OrdinaryDiffEq stores the two endpoint RHS values.

## Provenance and numerical behavior

Pinned revision: `211142263781255a9aa2f910f6760b9f18ec29c8`.
Coefficient sources are
`OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl`,
`OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`, and the value
formulas in `OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl`.

The exact nested upstream polynomial is evaluated from owned `H * stage`
increments. Method-specific extensions add no RHS calls. Hermite endpoint RHS
work is moved before sampling/root localization and reused by acceptance, so it
also adds no calls. Retention owns one segment per accepted step; in-solve
sampling remains allocation-free after workspace construction.

Rust covers all 40 public native/configured types for endpoint agreement,
forward/backward retained queries, representative convergence, pinned samples,
all three root-dispatch forms, callback discontinuities, and RHS counts.
Rodas6P and Rodas23W/3P pinned comparisons use smaller fixed steps because the
pre-existing coarse-step Rust kernels differ from Julia independently of dense
interpolation; no tolerance was weakened.

The pinned sample comparisons agree within `2e-9` absolute error and the
representative continuous-root times within `8e-11`. `save_at` states are
bitwise identical to queries of separately retained segments at the same
times.

## Commands

- `cargo test --test rosenbrock_dense`: 6 passed.
- Rosenbrock/Rodas allocation suites: 3 passed.
- `cargo test --all-targets --all-features`: 393 passed across 126 suites.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- pinned `julia.exe --project=tests/julia tests/julia/rosenbrock_dense.jl`:
  passed.

## Remaining gaps

No implemented Rosenbrock-family value interpolant remains unwired. Dense
derivative queries and DAE algebraic-variable masks are outside the current
regular-ODE public API. The latter matters only once mass-matrix/DAE problems
are added; upstream suppresses Hermite derivative corrections on algebraic
variables.
