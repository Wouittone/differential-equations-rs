# Anas5 parity handoff

- Upstream pin: `OrdinaryDiffEq.jl@211142263781255a9aa2f910f6760b9f18ec29c8`
- Upstream package: `OrdinaryDiffEqLowOrderRK`
- Algorithm declaration: `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:220-241`
- Tableau source: `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1221-1311`
- Fixed-step implementation: `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:416-470`

Anas5 is a fixed-step six-stage method with an endpoint FSAL derivative. Its
`a65` coefficient is recomputed from the pinned periodicity estimate `w` and
signed step size using the upstream Anastassi–Simos trigonometric formula;
`Anas5::default()` uses `w = 1`. The custom shared-driver kernel retains the
endpoint derivative for the next accepted step and invalidates it after a
callback state mutation. Dense `save_at`, backward stepping, and callback
termination use the shared trajectory semantics.

Focused Rust coverage exercises convergence, forward/backward `save_at`,
callback termination, and step-count-invariant allocations. Julia validation
is unavailable here because `julia` is not on `PATH`; retry the pinned and
full suites after installing Julia and restoring `tests/julia/Manifest.toml`.
