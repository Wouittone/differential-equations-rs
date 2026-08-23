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
- Implemented and detected in matched Julia tests: **264**.
- Implemented without a detected matched Julia test: **0**.
- Missing in-scope public names: **81**.
- Explicitly excluded public names: **4**.

Aliases are public parity obligations but do not require a second numerical kernel.

## Family status

| Family | In scope | Implemented + Julia-tested | Implemented, Julia test not detected | Missing Rust implementation |
| --- | ---: | ---: | ---: | ---: |
| Adams multistep | 13 | 12 | 0 | 1 |
| approximate-matrix-factorization wrapper | 1 | 0 | 0 | 1 |
| automatic/default composite | 2 | 2 | 0 | 0 |
| BDF and IMEX multistep | 15 | 9 | 0 | 6 |
| explicit Runge-Kutta | 2 | 2 | 0 | 0 |
| exponential Runge-Kutta | 17 | 0 | 0 | 17 |
| extrapolation | 7 | 0 | 0 | 7 |
| fully implicit Runge-Kutta | 5 | 0 | 0 | 5 |
| high-order explicit Runge-Kutta | 16 | 16 | 0 | 0 |
| IMEX multistep | 2 | 0 | 0 | 2 |
| linear and Lie-group methods | 18 | 0 | 0 | 18 |
| low-order explicit Runge-Kutta | 28 | 28 | 0 | 0 |
| low-storage explicit Runge-Kutta | 44 | 44 | 0 | 0 |
| multirate and MRI-GARK | 9 | 0 | 0 | 9 |
| Nordsieck variable-order multistep | 4 | 0 | 0 | 4 |
| parallel diagonally implicit Runge-Kutta | 1 | 0 | 0 | 1 |
| parallel explicit Runge-Kutta | 1 | 1 | 0 | 0 |
| QPRK explicit Runge-Kutta | 1 | 1 | 0 | 0 |
| Rosenbrock and Rosenbrock-W | 40 | 40 | 0 | 0 |
| Runge-Kutta interval prediction | 1 | 0 | 0 | 1 |
| Runge-Kutta-Nystrom | 17 | 17 | 0 | 0 |
| SDIRK, ESDIRK, and additive IMEX RK | 39 | 39 | 0 | 0 |
| second-order structural dynamics | 2 | 0 | 0 | 2 |
| SIMD explicit Runge-Kutta | 3 | 0 | 0 | 3 |
| stabilized explicit Runge-Kutta | 13 | 13 | 0 | 0 |
| stabilized implicit Runge-Kutta | 1 | 0 | 0 | 1 |
| strong-stability-preserving Runge-Kutta | 21 | 21 | 0 | 0 |
| symplectic and partitioned Runge-Kutta | 18 | 18 | 0 | 0 |
| Taylor series | 3 | 0 | 0 | 3 |
| user-tableau explicit Runge-Kutta | 1 | 1 | 0 | 0 |

## Missing Rust solver names by family

This is the implementation handoff list. Each entry remains in scope and lacks
a detected public Rust algorithm implementation. Required features and exact
upstream source locations are available in the JSON/CSV records.

### Adams multistep (1)

- `VCABM` — OrdinaryDiffEqAdamsBashforthMoulton; ODEProblem

### approximate-matrix-factorization wrapper (1)

- `AMF` — OrdinaryDiffEqAMF; ODEProblem with structured ODEFunction

### BDF and IMEX multistep (6)

- `IMEXEuler` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `IMEXEulerARK` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF2` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF3` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem
- `SBDF4` — OrdinaryDiffEqBDF; ODEProblem or SplitODEProblem

### exponential Runge-Kutta (17)

- `EPIRK4s3A` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK4s3B` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5P1` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5P2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EPIRK5s3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETD1` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETD2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK2` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `ETDRK4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exp4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exprb32` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `Exprb43` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `EXPRB53s3` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `HochOst4` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `LawsonEuler` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem
- `NorsettEuler` — OrdinaryDiffEqExponentialRK; ODEProblem or SplitODEProblem

### extrapolation (7)

- `AitkenNeville` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ExtrapolationMidpointDeuflhard` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ExtrapolationMidpointHairerWanner` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitDeuflhardExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitEulerBarycentricExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitEulerExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem
- `ImplicitHairerWannerExtrapolation` — OrdinaryDiffEqExtrapolation; ODEProblem

### fully implicit Runge-Kutta (5)

- `AdaptiveRadau` — OrdinaryDiffEqFIRK; ODEProblem
- `GaussLegendre` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA3` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA5` — OrdinaryDiffEqFIRK; ODEProblem
- `RadauIIA9` — OrdinaryDiffEqFIRK; ODEProblem

### IMEX multistep (2)

- `CNAB2` — OrdinaryDiffEqIMEXMultistep; ODEProblem or SplitODEProblem
- `CNLF2` — OrdinaryDiffEqIMEXMultistep; ODEProblem or SplitODEProblem

### linear and Lie-group methods (18)

- `CayleyEuler` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG2` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG3` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `CG4a` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LieEuler` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LieRK4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `LinearExponential` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusAdapt4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGauss4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL6` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusGL8` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusLeapfrog` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusMidpoint` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusNC6` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `MagnusNC8` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `RKMK2` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction
- `RKMK4` — OrdinaryDiffEqLinear; ODEProblem with linear/operator ODEFunction

### multirate and MRI-GARK (9)

- `MIS` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRAB` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MREEF` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK22a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK22b` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK33a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKERK45a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKESDIRK34a` — OrdinaryDiffEqMultirate; SplitODEProblem
- `MRIGARKIRK21a` — OrdinaryDiffEqMultirate; SplitODEProblem

### Nordsieck variable-order multistep (4)

- `AN5` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE_Adams` — OrdinaryDiffEqNordsieck; ODEProblem
- `JVODE_BDF` — OrdinaryDiffEqNordsieck; ODEProblem

### parallel diagonally implicit Runge-Kutta (1)

- `PDIRK44` — OrdinaryDiffEqPDIRK; ODEProblem

### Runge-Kutta interval prediction (1)

- `RKIP` — OrdinaryDiffEqRKIP; ODEProblem

### second-order structural dynamics (2)

- `GeneralizedAlpha` — OrdinaryDiffEqNewmark; SecondOrderODEProblem
- `NewmarkBeta` — OrdinaryDiffEqNewmark; SecondOrderODEProblem

### SIMD explicit Runge-Kutta (3)

- `MER5v2` — OrdinaryDiffEqSIMDRK; ODEProblem
- `MER6v2` — OrdinaryDiffEqSIMDRK; ODEProblem
- `RK6v4` — OrdinaryDiffEqSIMDRK; ODEProblem

### stabilized implicit Runge-Kutta (1)

- `IRKC` — OrdinaryDiffEqStabilizedIRK; ODEProblem

### Taylor series (3)

- `ExplicitTaylor` — OrdinaryDiffEqTaylorSeries; ODEProblem
- `ExplicitTaylor2` — OrdinaryDiffEqTaylorSeries; ODEProblem
- `ExplicitTaylorAdaptiveOrder` — OrdinaryDiffEqTaylorSeries; ODEProblem

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
