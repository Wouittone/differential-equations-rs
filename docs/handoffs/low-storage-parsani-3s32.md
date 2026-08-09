# ParsaniKetchesonDeconinck3S32 handoff

Implemented the fixed-step `ParsaniKetchesonDeconinck3S32` 3S low-storage Runge--Kutta method.

## Pinned source references

- `lib/OrdinaryDiffEqLowStorageRK/src/algorithms.jl`, revision `211142263781255a9aa2f910f6760b9f18ec29c8`, declaration and documentation for `ParsaniKetchesonDeconinck3S32`.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl`, `ParsaniKetchesonDeconinck3S32ConstantCache`, lines 704--730 in the pinned checkout.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl`, 3S constant-cache recurrence, lines 143--178 in the pinned checkout.

The Rust kernel preserves the pinned gamma/delta/beta/c stage recurrence, uses the shared fixed-step integration lifecycle, and evaluates the endpoint derivative for parity with the upstream FSAL bookkeeping. Adaptive stepping remains rejected as for the existing low-storage family.

## Validation

Focused Rust coverage includes design-order convergence, backward/save-at semantics, callback termination, malformed coefficient-shape validation, and one-step versus 1000-step allocation invariance. The Julia compliance fixture compares a non-autonomous endpoint against `ParsaniKetchesonDeconinck3S32()` from the pinned package.

Julia validation is pending because `julia` is not currently available on `PATH` in this environment. Retry `julia --project=tests/julia tests/julia/pinned_environment.jl --check` and `julia --project=tests/julia tests/julia/runtests.jl` after installing Julia and activating the pinned project.
