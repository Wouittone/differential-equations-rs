# MSRK6 wave handoff

Pinned upstream: `211142263781255a9aa2f910f6760b9f18ec29c8`.

Implemented `Msrk6` as a fixed-step regular ODE solver using the pinned
eight-stage tableau from
`lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl` and the
`MSRK6ConstantCache` perform-step definition in
`lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl`. The final
endpoint row and zero final weight represent OrdinaryDiffEqCore's default FSAL
lifecycle, including endpoint derivative reuse on the next accepted step.

Coverage includes sixth-order convergence, forward/backward integration,
`save_at`, callback termination, and one-step versus 1000-step allocation
invariance. Rust formatting, all-target tests, Clippy, and diff checks pass.

Julia pinned/full gates are pending because the coordinator environment has no
`julia` executable. Retry with:

```text
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```
