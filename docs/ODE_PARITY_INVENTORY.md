# Regular ODE parity inventory

This is a generated summary of native solver exports at SciML/OrdinaryDiffEq.jl
revision `211142263781255a9aa2f910f6760b9f18ec29c8` under the scope in
[UPSTREAM_SCOPE.md](UPSTREAM_SCOPE.md). Regenerate it with:

```powershell
./scripts/generate_ode_inventory.ps1 -UpstreamPath <path-to-OrdinaryDiffEq.jl>
```

The full machine-readable records, including upstream definition paths and line
numbers, problem representations, fixed/adaptive behavior, Jacobian, linear-solver,
dense-output and controller requirements, aliases, exclusions, and current Rust and
Julia status, are in [ode_algorithm_inventory.json](ode_algorithm_inventory.json)
and [ode_algorithm_inventory.csv](ode_algorithm_inventory.csv).

## Totals

- Public solver names inspected: **349**.
- In-scope regular ODE names: **345**
  (333 canonical/composite constructors and
  12 public aliases).
- Implemented and detected in matched Julia tests: **345**.
- Implemented without a detected matched Julia test: **0**.
- Missing in-scope public names: **0**.
- Explicitly excluded public names: **4**.

Aliases are public parity obligations but do not require a second numerical kernel.

## Family status

| Family | In scope | Implemented + Julia-tested | Implemented, Julia test not detected | Missing Rust implementation |
| --- | ---: | ---: | ---: | ---: |
| Adams multistep | 13 | 13 | 0 | 0 |
| approximate-matrix-factorization wrapper | 1 | 1 | 0 | 0 |
| automatic/default composite | 2 | 2 | 0 | 0 |
| BDF and IMEX multistep | 15 | 15 | 0 | 0 |
| explicit Runge-Kutta | 2 | 2 | 0 | 0 |
| exponential Runge-Kutta | 17 | 17 | 0 | 0 |
| extrapolation | 7 | 7 | 0 | 0 |
| fully implicit Runge-Kutta | 5 | 5 | 0 | 0 |
| high-order explicit Runge-Kutta | 16 | 16 | 0 | 0 |
| IMEX multistep | 2 | 2 | 0 | 0 |
| linear and Lie-group methods | 18 | 18 | 0 | 0 |
| low-order explicit Runge-Kutta | 28 | 28 | 0 | 0 |
| low-storage explicit Runge-Kutta | 44 | 44 | 0 | 0 |
| multirate and MRI-GARK | 9 | 9 | 0 | 0 |
| Nordsieck variable-order multistep | 4 | 4 | 0 | 0 |
| parallel diagonally implicit Runge-Kutta | 1 | 1 | 0 | 0 |
| parallel explicit Runge-Kutta | 1 | 1 | 0 | 0 |
| QPRK explicit Runge-Kutta | 1 | 1 | 0 | 0 |
| Rosenbrock and Rosenbrock-W | 40 | 40 | 0 | 0 |
| Runge-Kutta interval prediction | 1 | 1 | 0 | 0 |
| Runge-Kutta-Nystrom | 17 | 17 | 0 | 0 |
| SDIRK, ESDIRK, and additive IMEX RK | 39 | 39 | 0 | 0 |
| second-order structural dynamics | 2 | 2 | 0 | 0 |
| SIMD explicit Runge-Kutta | 3 | 3 | 0 | 0 |
| stabilized explicit Runge-Kutta | 13 | 13 | 0 | 0 |
| stabilized implicit Runge-Kutta | 1 | 1 | 0 | 0 |
| strong-stability-preserving Runge-Kutta | 21 | 21 | 0 | 0 |
| symplectic and partitioned Runge-Kutta | 18 | 18 | 0 | 0 |
| Taylor series | 3 | 3 | 0 | 0 |
| user-tableau explicit Runge-Kutta | 1 | 1 | 0 | 0 |

## Missing Rust solver names by family

This is the implementation handoff list. Each entry remains in scope and lacks
a detected public Rust algorithm implementation. Required features and exact
upstream source locations are available in the JSON/CSV records.



## Rust implementations without detected Julia compliance

These public Rust implementations remain outside the Julia-tested coverage
count until a matched compliance invocation is detected.



## Aliases

| Public name | Kind | Canonical target | Package |
| --- | --- | --- | --- |
| `ETD1` | exact-alias | `NorsettEuler` | OrdinaryDiffEqExponentialRK |
| `IMEXEuler` | configured-alias | `SBDF(order=1)` | OrdinaryDiffEqBDF |
| `IMEXEulerARK` | configured-alias | `SBDF(order=1, ark=true)` | OrdinaryDiffEqBDF |
| `JVODE_Adams` | configured-alias | `JVODE(:Adams)` | OrdinaryDiffEqNordsieck |
| `JVODE_BDF` | configured-alias | `JVODE(:BDF)` | OrdinaryDiffEqNordsieck |
| `QBDF` | configured-alias | `QNDF(kappa=(0,0,0,0,0))` | OrdinaryDiffEqBDF |
| `QBDF1` | configured-alias | `QNDF1(kappa=0)` | OrdinaryDiffEqBDF |
| `QBDF2` | configured-alias | `QNDF2(kappa=0)` | OrdinaryDiffEqBDF |
| `SBDF2` | configured-alias | `SBDF(order=2)` | OrdinaryDiffEqBDF |
| `SBDF3` | configured-alias | `SBDF(order=3)` | OrdinaryDiffEqBDF |
| `SBDF4` | configured-alias | `SBDF(order=4)` | OrdinaryDiffEqBDF |
| `Tsit5DA` | configured-alias | `HybridExplicitImplicitRK(Tsit5DATableau, order=5)` | OrdinaryDiffEqRosenbrock |

## Explicit exclusions

| Public name | Package | Rationale |
| --- | --- | --- |
| `DABDF2` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `DFBDF` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `DImplicitEuler` | OrdinaryDiffEqBDF | DAE residual-form algorithm; DAE-only behavior is outside regular ODE scope. |
| `FunctionMap` | OrdinaryDiffEqFunctionMap | Discrete dynamical-system map, not a continuous initial-value ODE solver. |

Package-level exclusions from [UPSTREAM_SCOPE.md](UPSTREAM_SCOPE.md), such as
DelayDiffEq, StochasticDiffEq, external wrappers, BVP, PDE, and steady-state
solvers, are not expanded into per-algorithm rows because they are not part of
the OrdinaryDiffEq native ODE solver export surface.

## Classified support-only subpackages

| Package | Why it has no solver rows |
| --- | --- |
| `OrdinaryDiffEqCore` | Shared integrator abstractions and internal composite types; no public numerical method constructors. |
| `OrdinaryDiffEqDifferentiation` | Jacobian, time-derivative, and differentiation support used by solver packages. |
| `OrdinaryDiffEqExplicitTableaus` | Butcher-tableau data and constructors, not solver algorithms. |
| `OrdinaryDiffEqImplicitTableaus` | Implicit tableau data and constructors, not solver algorithms. |
| `OrdinaryDiffEqNonlinearSolve` | Nonlinear-solver and DAE-initialization support, not time-integration algorithms. |
| `OrdinaryDiffEqRosenbrockTableaus` | Rosenbrock tableau data and constructors, not solver algorithms. |

## Audited non-algorithm exports

| Export | Package | Why it has no solver row |
| --- | --- | --- |
| `Predictor` | `OrdinaryDiffEqSDIRK` | Nonlinear-stage predictor enum namespace, not an ODE algorithm constructor. |

## Interpretation notes and uncertainties

- The inventory treats package exports as the public algorithm surface; internal, unexported experimental types are not parity targets.
- ETD1 is the only exact exported type alias found at the pinned revision. Named functions that return a configured canonical algorithm are recorded as configured aliases. Auto* and Default* names are composite constructors.
- AMF is counted as a native wrapper constructor because it dispatches back into OrdinaryDiffEq with native Rosenbrock-W methods.
- Rust status requires both a public crate export and a concrete implementation of an algorithm trait. Compatibility aliases to substitute kernels do not count as implementations.
- Julia compliance requires an imported OrdinaryDiffEq constructor to be invoked outside its using block; imports alone do not count as tests. Numerical assertion quality remains a review concern.
- Implemented status measures public algorithm-name coverage only. It does not establish parity for every upstream problem representation or shared feature; consult problem_representation, required_features, and FEATURE_COVERAGE.md separately.
- Nonsingular mass-matrix behavior of dual ODE/DAE methods is included, while residual-form DAE constructors and singular-mass-matrix behavior are excluded.
