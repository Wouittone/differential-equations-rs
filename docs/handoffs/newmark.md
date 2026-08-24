# Newmark structural-method handoff

## Scope

This wave implements `NewmarkBeta` and `GeneralizedAlpha` for the crate's
typed `SecondOrderOdeProblem` representation. The implementation follows
`OrdinaryDiffEqNewmark` at pinned revision
`211142263781255a9aa2f910f6760b9f18ec29c8`.

The upstream parameter validation and constructors are in
`lib/OrdinaryDiffEqNewmark/src/algorithms.jl`. The Newmark update,
generalized-alpha blending, and nonlinear acceleration residual are in
`newmark_perform_step.jl` and `newmark_nlsolve.jl`.

## Numerical implementation

For an acceleration candidate `a_(n+1)`, the kernel constructs

- `v_(n+1) = v_n + h*((1-gamma)*a_n + gamma*a_(n+1))`;
- `q_(n+1) = q_n + h*v_n + h^2/2*((1-2*beta)*a_n + 2*beta*a_(n+1))`.

It then solves the pinned generalized-alpha residual at the blended
`alpha_m`/`alpha_f` acceleration, velocity, position, and time using a dense
finite-difference Newton system. `alpha_m = alpha_f = 0` is the Newmark case.
Fixed stepping uses one structural step. Adaptive stepping uses a full step
versus two half steps, accepting the two-half-step result and preserving the
shared rejection/callback lifecycle.

`GeneralizedAlpha` exposes the direct four-parameter constructor plus the
pinned spectral-radius and HHT parameterizations. `NewmarkBeta` defaults to
`beta = 1/4`, `gamma = 1/2`.

## Verification

Rust tests cover second-order convergence, both constructor
parameterizations, high-frequency algorithmic damping, adaptive integration,
backward integration, continuous termination, statistics, and invalid
parameters. The matched Julia fixture compares both public constructors on
the same fixed-step harmonic oscillator. The focused Julia comparison and
pinned-environment check pass with `OrdinaryDiffEqNewmark` sourced directly
from the reference monorepo revision.

## Limitation

The current `SecondOrderOdeProblem` represents `q'' = f(v, q, p, t)` with an
identity mass operator. OrdinaryDiffEqNewmark also supports nonsingular mass
matrices. That representation and solve path remain part of the broader mass
matrix feature work; the identity-mass methods here are not claimed to cover
that option.
