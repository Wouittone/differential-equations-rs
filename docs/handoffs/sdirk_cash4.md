# Cash4 SDIRK handoff

## Summary

Added a native regular first-order Cash4 SDIRK solver with a five-stage
Newton kernel, analytic or finite-difference Jacobians, adaptive embedding,
fixed-step operation, backward integration, callbacks, and `save_at` support
through the shared driver.

## Files changed

- `src/sdirk_cash4.rs`
- `src/lib.rs`
- `tests/sdirk_cash4.rs`
- `examples/sdirk_cash4_compliance.rs`
- `tests/julia/sdirk_cash4.jl`

## Public API

- `differential_equations::algorithms::implicit::Cash4`

## Upstream source and revision

- `lib/OrdinaryDiffEqSDIRK/src/sdirk_tableaus.jl`, `Cash4Tableau`, lines
  319--407
- `lib/OrdinaryDiffEqSDIRK/src/imex_tableaus.jl`, `Cash4ESDIRKIMEXTableau`,
  lines 2528--2561 (used to confirm the stiffly-accurate primary weights and
  default `embedding = 3` estimator)
- pinned revision `211142263781255a9aa2f910f6760b9f18ec29c8`

## Numerical differences and limitations

The Rust implementation targets regular `OdeProblem` only. Split/IMEX,
mass-matrix, custom nonlinear solver, predictor, and smoothing options remain
outside this task. Jacobians are rebuilt per stage because Cash4 has distinct
stage times and a nonautonomous Jacobian must not reuse a factorization from a
different time.

## Verification

Run `cargo fmt -- --check`, `cargo test --all-targets`,
`cargo clippy --all-targets -- -D warnings`, and `git diff --check`. The Julia
fixture is ready for inclusion by the coordinator; local Julia availability
must be checked by the coordinator.
