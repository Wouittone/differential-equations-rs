# ParsaniKetchesonDeconinck3S94 handoff

Implemented the fixed-step `ParsaniKetchesonDeconinck3S94` 3S low-storage
Runge--Kutta method using the integrated 3S kernel.

## Pinned source references

- `lib/OrdinaryDiffEqLowStorageRK/src/algorithms.jl`, revision
  `211142263781255a9aa2f910f6760b9f18ec29c8`, declaration and documentation
  for `ParsaniKetchesonDeconinck3S94`.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl`,
  `ParsaniKetchesonDeconinck3S94ConstantCache`, lines 964--1028 in the pinned
  checkout.
- `lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl`, shared
  3S constant-cache recurrence, lines 143--178 in the pinned checkout.

The Rust constructor preserves all eight gamma/delta/beta/c stage coefficients
and uses the shared fixed-step lifecycle, callback handling, save-at recording,
and allocation-invariant 3S cache. Adaptive stepping remains unsupported for
this fixed-only method.

Focused Rust coverage includes fourth-order convergence, backward/save-at
semantics, callback termination, malformed 3S shape validation, and one-step
versus 1000-step allocation invariance. The Julia fixture compares a
non-autonomous endpoint against `ParsaniKetchesonDeconinck3S94()` from the
pinned package.

Julia validation is pending because no executable named `julia` is available
on this worker. Retry `julia --project=tests/julia
tests/julia/pinned_environment.jl --check` and
`julia --project=tests/julia tests/julia/runtests.jl` after installing a real
Julia executable and activating the pinned project.
