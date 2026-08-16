# KYKSSPRK42 handoff

Summary: Ported the pinned Kubatko--Yeager--Ketcheson optimal SSPRK(4,2)
method as a fixed-step regular ODE algorithm.  The Shu--Osher recurrence is
expanded into an equivalent four-stage explicit Butcher tableau for reuse of
the native driver.

Files changed:

- `src/ssprk_kyk42.rs`
- `src/lib.rs`
- `examples/ssprk_compliance.rs`
- `tests/julia/ssprk.jl`

Public APIs added: `KykSsprk42` and the upstream-spelling alias `KYKSSPRK42`.

Upstream source and revision: `OrdinaryDiffEqSSPRK/src/algorithms.jl`,
`ssprk_caches.jl`, and `ssprk_perform_step.jl` at
`211142263781255a9aa2f910f6760b9f18ec29c8`.

Rust tests: second-order convergence, fixed-only adaptive rejection,
backward integration with `save_at`, and terminating continuous callback.

Julia tests: added the fixed-step exponential endpoint comparison to
`tests/julia/ssprk.jl`; the coordinator must retain the aggregate include.

Commands run: pending final handoff verification; see the coordinator report
for the exact gate results.

Numerical differences: the native driver does not expose upstream stage and
step limiter callbacks.  It computes the endpoint state from the same
expanded recurrence and uses its standard dense Hermite segment behavior.

Allocation/performance impact: uses the existing explicit RK workspace and
does not add per-step allocations beyond the shared driver.

Known limitations: the pinned algorithm has no embedded estimator and is
therefore fixed-step only; `adaptive = true` returns
`SolveError::AdaptiveStepUnsupported`.

Follow-up dependencies: regenerate the regular-ODE inventory after merging;
the coordinator owns the inventory and status documents.

Recommended next task: run the full Julia compliance suite and final solver
family audit after merging this branch.
