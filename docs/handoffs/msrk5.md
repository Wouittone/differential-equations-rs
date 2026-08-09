# MSRK5 handoff

## Scope

Implemented the regular fixed-step `MSRK5` algorithm from the pinned
OrdinaryDiffEq.jl revision `211142263781255a9aa2f910f6760b9f18ec29c8`.
The source references are `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:324`,
`low_order_rk_tableaus.jl:1477-1561`, and
`low_order_rk_perform_step.jl:1425-1460`.

## Implementation

`Msrk5` is exposed from `src/lib.rs` and uses the shared explicit tableau
kernel. The nine-stage tableau includes the endpoint derivative as an FSAL
stage: stage 9 has `c = 1`, its first eight coefficients equal the update
weights, and its ninth weight is zero. This preserves the upstream eight RHS
evaluations per accepted step after initialization while retaining callback
invalidation semantics from the shared driver.

The compliance example and Julia fixture are extended with the `msrk5` key.
Rust coverage includes fifth-order convergence, forward/backward `save_at`,
continuous callback termination, and callback-free allocation invariance.

## Validation

Passing in the isolated worktree:

- `cargo fmt -- --check`
- `cargo test --all-targets` (95 library tests and all integration/examples)
- `cargo clippy --all-targets -- -D warnings`
- `git diff --check`

Julia validation could not run because no `julia` executable is installed or
on `PATH` (`julia: The term 'julia' is not recognized`). Retry the pinned and
full Julia suites after installing Julia and restoring the pinned test
environment.

Commit: `da67b51bba32d7eef4e497e39caf2c57bfb4f284`.
