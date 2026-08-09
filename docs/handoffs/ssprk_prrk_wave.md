# Parametric relaxation SSPRK22 wave

This wave adds the native regular-ODE `Prrk22` constructor (also exported under
the upstream-compatible `pRRK22` alias). It is fixed-step only, as in the
pinned OrdinaryDiffEqSSPRK algorithm declaration. The default `kappa = 0`
reduces exactly to SSPRK22; nonzero `kappa` values use OrdinaryDiffEq's
per-step coefficient rescaling and effective timestep.

Pinned source: OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`,
`lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:270-289` (pRRK22 declaration) and
`lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:1481-1566` (coefficient
rescaling helper and constant/mutable perform-step paths). The Rust kernel
implements the same `psi` recurrence, modified alpha/beta coefficients,
abscissae, and `dt_hat = c_hat2 * dt` semantics while retaining the shared
callback, backward, and accepted-step lifecycle.

The method intentionally does not claim adaptive stepping or OrdinaryDiffEq's
stage/step limiter threading wrappers. Focused tests cover fixed-step order,
nonzero relaxation, backward integration, callback termination, and allocation
invariance.
