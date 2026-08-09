# SSPRK432 handoff

## Scope

This wave adds the native regular ODE `SspRk432` facade and exports it from the
crate. It reuses the checked explicit RK kernel and therefore supports both
fixed-step and adaptive integration, backward spans, callbacks, `save_at`, and
the shared endpoint Hermite dense path. The callback-free fixed-step workspace
has a step-count-invariant allocation shape (`tests/ssprk432_allocations.rs`).

## Pinned upstream evidence

The implementation is matched to OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

* `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:105-119` — adaptive-capable
  `SSPRK432` declaration and order 3 metadata.
* `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:851-946` — four-stage
  Shu–Osher sequence and the `(1/3, 1/3, 1/3, 0)` embedded estimate.
* `lib/OrdinaryDiffEqSSPRK/src/interpolants.jl:26-198` — SSPRK shared dense
  interpolant dispatch (the Rust shared Hermite service covers the same
  endpoint/save-at contract).

The Rust tableau is
`A = [[0], [1/2], [1/2,1/2], [1/6,1/6,1/6]]`,
`b = [1/6,1/6,1/6,1/2]`, and the full embedded high-minus-low residual
`e = [-1/6,-1/6,-1/6,1/2]`. This differs intentionally from `SspRk43`,
which scales the same residual by one half.

## Validation

Passed on the isolated branch:

* `cargo fmt -- --check`
* `cargo test --all-targets` (93 library tests and all integration/example targets)
* `cargo clippy --all-targets -- -D warnings`
* `git diff --check`
* focused SSPRK432 fixed/adaptive order, backward/save-at, callback, and
  allocation-invariance tests

The Julia executable was unavailable on this worker (`julia` is not on PATH),
so the pinned-environment check and full `tests/julia/runtests.jl` must be run
from a coordinator environment that has Julia installed. The compliance
example emits `ssprk432,2.71828182765046700e0`; compare this with
`SSPRK432()` at `abstol=reltol=1e-9` in the pinned fixture when integrating.
