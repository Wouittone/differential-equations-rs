# Remaining Adams/BDF/IMEX multistep parity

## Summary

Implements the adaptive-order `VCABM` Adams--Moulton method and the fixed-step
split methods `IMEXEuler`, `IMEXEulerARK`, configurable `SBDF`, `SBDF2`,
`SBDF3`, `SBDF4`, `CNAB2`, and `CNLF2`. Split methods retain explicit and
implicit right-hand sides, solve only the implicit component with Newton's
method, and support either an analytic implicit Jacobian or finite differences.

## Public APIs added

- `algorithms::multistep::{Vcabm, VCABM}`
- `algorithms::multistep::{Sbdf, SBDF}` with `Sbdf::new(order)` for orders 1--4
- `algorithms::multistep::{ImexEuler, IMEXEuler, ImexEulerArk, IMEXEulerARK}`
- `algorithms::multistep::{Sbdf2, SBDF2, Sbdf3, SBDF3, Sbdf4, SBDF4}`
- `algorithms::multistep::{Cnab2, CNAB2, Cnlf2, CNLF2}`
- `solve_split`
- `SplitOdeProblem::with_implicit_jacobian`

## Upstream source and revision

Pinned to SciML/OrdinaryDiffEq.jl revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/algorithms.jl`
- `lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl`
- `lib/OrdinaryDiffEqBDF/src/algorithms.jl`
- `lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl`
- `lib/OrdinaryDiffEqIMEXMultistep/src/algorithms.jl`
- `lib/OrdinaryDiffEqIMEXMultistep/src/imex_multistep_perform_step.jl`

## Rust tests

`tests/multistep_remaining.rs` covers all public constructors, configured/named
alias equivalence, scalar/vector/nonautonomous problems, split staging,
forward/backward integration, refinement, failures, Jacobian paths, and solver
statistics.

## Julia tests

`tests/julia/multistep_remaining.jl` invokes all nine public constructors through
the Rust compliance fixture and compares endpoints with their pinned Julia
counterparts and independent analytic references.

## Numerical differences

The formulas and startup sequence match the pinned regular split-ODE paths.
`CNLF2` inherits the pinned first-order IMEX-Euler startup, so that startup can
dominate short end-to-end refinement sequences even though the leapfrog
recurrence is second order. `VCABM` uses the same variable-step interpolatory
Adams predictor/corrector family and adaptive order range (one through twelve),
with the crate's shared proportional controller instead of Julia's DDEABM
controller internals.

## Allocation/performance impact

Both kernels preallocate state, derivative, Newton, and history workspaces.
Accepted-step history rotation and nonlinear iterations do not allocate with
fixed dimension. Finite-difference Jacobians require one implicit-RHS evaluation
per state component; analytic implicit Jacobians avoid those evaluations.

## Known limitations

- Residual DAEs and singular mass matrices remain out of scope.
- Julia's `autodiff`, `concrete_jac`, `linsolve`, `nlsolve`, `kappa`, `tol`,
  `extrapolant`, and internal broadcast-threading keywords are not represented.
- Split problems do not yet expose callback registration.
- These multistep split methods require fixed stepping.

## Follow-up dependencies

None for regular typed split ODEs. A future DAE problem representation and
pluggable nonlinear/linear solver interfaces are prerequisites for the omitted
upstream options.
