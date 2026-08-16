# KYK2014DGSSPRK_3S2 handoff

`Kyk2014DgSsprk3S2` ports the fixed-step `KYK2014DGSSPRK_3S2` method from
OrdinaryDiffEqSSPRK at revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

The coefficients are copied from
`lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl` (the constant cache) and the
recurrence in `ssprk_perform_step.jl`.  The native tableau is the exact
algebraic expansion of the three-stage Shu--Osher update, with stage times
`0`, `β₁₀`, and `α₂₁β₁₀ + β₂₁`.  The upstream metadata reports order 2 and SSP
coefficient `0.8417`; this facade is fixed-step and rejects adaptive options.

The shared explicit RK driver supplies backward integration, `save_at`, dense
sampling, and continuous callback termination.  It does not expose Julia's
stage/step limiter callback hooks, so those upstream options remain outside the
regular `OdeProblem` API.  The upstream implementation evaluates an endpoint
derivative for interpolation and FSAL bookkeeping; the shared driver instead
uses its standard Hermite dense-output path and does not claim FSAL.

Validation in this wave includes second-order convergence, backward/save-at,
callback termination, allocation invariance, Rust compliance output, and a
Julia `ssprk.jl` reference row.
