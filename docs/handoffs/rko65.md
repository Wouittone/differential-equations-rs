# RKO65 parity handoff

## Scope

This wave adds the regular fixed-step `Rko65` constructor, the Rust tests and
low-order compliance fixture needed to exercise it, and the pinned Julia
fixture entry. The implementation targets OrdinaryDiffEqLowOrderRK at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

Upstream references:

- `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:244-252`
- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:698-769`
- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:809-839`
- `lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_classic.jl:326-368`

## Representation

The published tableau has `c₁ = 2/3`, so its first stage is not the derivative
at the left endpoint. The shared explicit driver intentionally validates and
reuses a `c=0` first stage for endpoint Hermite dense output. `RKO65` therefore
stores an additional unweighted `f(uₙ,tₙ)` stage at index zero and shifts the
six upstream stages to indices one through six. The added row has zero weight,
and all upstream coefficients, nodes, and update weights are copied exactly.
This preserves the method's fifth-order update while making `save_at`, backward
integration, and callback lifecycle semantics safe under the shared driver.

## Validation

- `cargo fmt -- --check` passed.
- `cargo test --all-targets` passed (97 library tests plus all integration and
  example targets, including RKO65 convergence, callbacks, save-at, and
  allocation tests).
- `cargo clippy --all-targets -- -D warnings` passed.
- `git diff --check` passed.
- Release low-order compliance fixture produced
  `rko65,2.71828182845895094e0` for `u'=u`, `dt=0.01`.
- Julia could not be run in this worktree because the `julia` executable is not
  installed. Retry `julia --project=tests/julia tests/julia/runtests.jl` once
  Julia is available; the fixture includes `RKO65()` at `dt=0.01`.

## Definition of done

`Rko65` is publicly exported, uses the exact pinned tableau, rejects adaptive
configuration like the upstream fixed-step algorithm, and has forward,
backward, fifth-order convergence, callback termination, save-at, and bounded
allocation coverage.

