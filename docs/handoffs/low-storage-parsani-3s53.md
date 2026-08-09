# ParsaniKetchesonDeconinck3S53 handoff

Implemented the fixed-step `ParsaniKetchesonDeconinck3S53` 3S low-storage
Runge--Kutta method using the existing validated 3S recurrence kernel.

## Pinned upstream evidence

- `lib/OrdinaryDiffEqLowStorageRK/src/algorithms.jl`, revision
  `211142263781255a9aa2f910f6760b9f18ec29c8`, declares the five-stage,
  third-order `ParsaniKetchesonDeconinck3S53` algorithm at lines 238--252.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl`, same revision,
  defines `ParsaniKetchesonDeconinck3S53ConstantCache` at lines 792--830.
  The Rust coefficient arrays preserve the pinned decimal literals and stage
  times exactly, with four loop entries plus the initial `β1` stage.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl`, same
  revision, `LowStorageRK3SConstantCache` initialization and perform-step
  recurrence at lines 144--177. The integrated Rust kernel follows the same
  `u1`, `tmp += δ*u`, `u = γ1*u + γ2*tmp + γ3*uprev + β2*dt*k`, and endpoint
  derivative sequence.

## Rust coverage

- Exported the public constructor from `src/lib.rs`.
- Added design-order, backward/save-at, discrete callback termination, and
  malformed-shape coverage alongside the existing 3S tests.
- Added one-step versus 1000-step allocation invariance coverage.
- Added the compliance example output and Julia fixture entry.

## Validation

- `cargo fmt -- --check` passed.
- `cargo test --all-targets` passed (95 library tests and all integration and
  example targets).
- `cargo clippy --all-targets -- -D warnings` passed.
- `git diff --check` passed.
- The Julia executable is not installed on this worker (`julia` command not
  found), so pinned-environment and full Julia compliance checks require the
  documented environment retry.
