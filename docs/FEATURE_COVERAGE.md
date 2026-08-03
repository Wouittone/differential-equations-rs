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
- matched OrdinaryDiffEq.jl compliance cases for discrete state effects,
  continuous termination, and `saveat` behavior.

## Remaining feature work

- method-specific high-order dense interpolation; current continuous-event
  localization and `save_at` sampling use linear interpolation inside an
  accepted step;
- Julia's full integrator callback interface, including parameter and step-size
  mutation, `save_positions`, callback initialization/finalization hooks,
  callback sets, vector continuous callbacks, preset-time stops, and the
  prebuilt DiffEqCallbacks.jl library;
- out-of-place functions, arbitrary scalar/state container types, mass
  matrices, split problems, general partitioned/dynamical problems, implicit
  second-order structural problems, and sensitivities.

Boundary-value conditions are not an initial-value ODE feature and remain
excluded by [`UPSTREAM_SCOPE.md`](UPSTREAM_SCOPE.md). Adding them would require
a separate BVP problem and solver API rather than a callback option.
