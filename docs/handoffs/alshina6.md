# Alshina6 parity handoff

- Upstream pin: `OrdinaryDiffEq.jl@211142263781255a9aa2f910f6760b9f18ec29c8`
- Upstream package: `OrdinaryDiffEqLowOrderRK`
- Algorithm declaration: `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:491-512`
- Tableau source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1884-1965`
- Step source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:2167-2208`

The Rust `Alshina6` facade uses the shared explicit Runge–Kutta step kernel with
the seven-stage, sixth-order tableau copied from the pinned constant cache. It
is fixed-step only, has no embedded error estimate, and preserves shared
callback, backward integration, and `save_at` behavior. The runtime weights
are `[1/12, 0, 0, 0, 5/12, 5/12, 1/12]`, matching the upstream update that
omits stages 2–4.

Focused coverage includes exponential convergence, callback termination,
forward/backward `save_at` samples, and callback-free allocation invariance.
The low-order Julia fixture includes `Alshina6`; the Julia pinned/full gates
could not run in this environment because `julia` is not installed or on
`PATH`. Retry those commands after installing Julia and restoring the pinned
environment from `tests/julia/Project.toml` and `tests/julia/Manifest.toml`.
