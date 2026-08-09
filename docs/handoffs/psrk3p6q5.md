# PSRK3p6q5 parity handoff

Implemented the native regular fixed-step `Psrk3p6q5` explicit Runge--Kutta
algorithm from OrdinaryDiffEqLowOrderRK revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

## Upstream evidence

- Algorithm declaration: `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:386-404`.
- Exact constant tableau: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1405-1433`.
- Upstream fixed-step implementation evaluates five stages with the listed
  `a21..a54`, `b1..b5`, and `c2..c5`; the shared explicit driver performs the
  same lower-triangular update and endpoint evaluation.
- Upstream `alg_order(PSRK3p6q5()) = 3` and `isfsal = false`.

## Rust surface

- Added `Psrk3p6q5` and its exact five-stage third-order tableau to
  `src/explicit_rk.rs` and exported it from `src/lib.rs`.
- Added fixed-step convergence, forward/backward `save_at`, callback
  termination, adaptive rejection, and callback-free allocation tests.
- Added the fixed-step compliance endpoint to `examples/low_order_compliance.rs`
  and the pinned Julia low-order fixture.

## Verification

- `cargo fmt -- --check` passes.
- Targeted `cargo test --test psrk3p6q5 --test psrk3p6q5_allocations` passes
  (5 tests).
- Full Rust gates and pinned Julia/full suite should be rerun by the
  coordinator after cherry-pick. The Julia executable was unavailable in the
  worker environment when checked.
