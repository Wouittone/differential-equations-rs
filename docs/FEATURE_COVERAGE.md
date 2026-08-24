# ODE feature coverage

Algorithm-name coverage is tracked separately in
[`ALGORITHM_COVERAGE.md`](ALGORITHM_COVERAGE.md). This page describes the
problem, callback, and output APIs available to the implemented Rust
algorithms.

## Implemented

- in-place first-order ODE right-hand sides with vector `f64` state;
- arbitrary vector initial conditions, parameters, forward or backward time
  spans, and optional dense analytic state Jacobians;
- a separate in-place `SecondOrderOdeProblem` API for the `q' = v`,
  `v' = f(v, q, p, t)` specialization, with separate position and velocity
  storage; this is not yet a general `DynamicalODEProblem` with an independent
  position-rate function;
- discrete callbacks evaluated at initialization and after accepted steps;
- scalar continuous callbacks with rising, falling, or bidirectional crossing
  filters;
- callback state mutation, integration continuation, and termination;
- automatic invalidation of FSAL derivatives, Adams history, implicit
  factorizations, and Rosenbrock differentiation data after callback effects;
- ordered `save_at` sampling in either integration direction;
- opt-in retained dense segments and post-solve [`Solution::interpolate`]
  queries for shared explicit Runge--Kutta solvers; endpoint Hermite segments
  are used when no special extension is available;
- pinned method-specific continuous extensions for Tsit5, DP5, and
  Owren--Zennaro 3/4/5, shared consistently by `save_at`, scalar continuous
  event localization, and retained post-solve queries;
- the pinned DPRKN6 continuous extension for in-solve second-order sampling and
  scalar continuous event localization;
- implicit identity-mass structural dynamics through adaptive/fixed
  Newmark--beta and generalized-alpha methods;
- matched OrdinaryDiffEq.jl compliance cases for discrete state effects,
  continuous termination, and `saveat` behavior.

## Remaining feature work

- method-specific high-order dense interpolation for the remaining families.
  BS5 and Verner require their upstream lazy extra interpolation stages;
  Rosenbrock/Rodas require retained stiff stage combinations; SSP methods need
  their per-method polynomial dispatch; and multistep, SDIRK/TRBDF2,
  stabilized, split, symplectic, and remaining second-order methods still need
  family-specific accepted-segment hooks. Solvers without such a hook retain
  their existing endpoint fallback and must not be described as having a
  method-specific high-order extension;
- Julia's full integrator callback interface, including parameter and step-size
  mutation, `save_positions`, callback initialization/finalization hooks,
  callback sets, vector continuous callbacks, preset-time stops, and the
  prebuilt DiffEqCallbacks.jl library;
- out-of-place functions, arbitrary scalar/state container types, mass
  matrices, general partitioned/dynamical position-rate functions, structural
  problems with non-identity mass matrices, and sensitivities.

Boundary-value conditions are not an initial-value ODE feature and remain
excluded by [`UPSTREAM_SCOPE.md`](UPSTREAM_SCOPE.md). Adding them would require
a separate BVP problem and solver API rather than a callback option.
