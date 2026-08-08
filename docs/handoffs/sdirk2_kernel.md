# SDIRK2 kernel handoff

## Scope

This wave adds the regular, non-split two-stage Sdirk2 algorithm from the
pinned OrdinaryDiffEqSDIRK revision 211142263781255a9aa2f910f6760b9f18ec29c8.
The implementation uses the frozen first-order driver and checked dense
Jacobian/LU interfaces. Split/IMEX problems, singular mass matrices, and DAE
residual forms are intentionally excluded.

The generated tableau is the exact pinned record:

* stage times (1, 0);
* A = [[1, 0], [-1, 1]];
* primary weights (1/2, 1/2);
* embedded weights (1/2, -1/2).

Each stage is solved by Newton iteration with finite-difference fallback when
no analytic Jacobian is supplied. The unit diagonal permits factorization
reuse across both stages of an attempt. The embedded estimate is scaled by
the driver's absolute/relative tolerances and drives adaptive rejection.

## Validation

The branch passed cargo test --all-targets; this includes scalar stiff decay,
second-order convergence, nonautonomous and backward integration, analytic and
finite-difference Jacobian parity, adaptive rejection, and callback
termination tests in tests/sdirk2.rs. The sdirk2_compliance example emits a
deterministic endpoint suitable for the pinned Julia SDIRK2() reference. The
coordinator-added `tests/julia/sdirk2.jl` compares the fixed `dt=0.01`
endpoint against pinned `OrdinaryDiffEqSDIRK.SDIRK2()` and checks adaptive
termination statistics; the full suite passes 206 assertions across 16
testsets.

src/lib.rs was temporarily exported on this isolated branch so the
integration tests could run. The coordinator should retain the mod sdirk;
and pub use sdirk::Sdirk2; export when merging.

## Known parity caveat

The Rust shared controller currently exposes proportional metadata only; this
wave uses its order-two proportional policy. OrdinaryDiffEq's generic
PIController has algorithm-specific history gains, so adaptive step counts
may differ even when endpoint values meet tolerance. Fixed-step endpoint
parity and convergence are the stable compliance anchors for this wave.
