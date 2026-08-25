# differential-equations-rs

A beta Rust port of the ODE solvers in Julia's
[DifferentialEquations.jl ecosystem](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).

## Goals

This project exists first and foremost as a proof of concept to answer a
specific performance question:

> Can Rust ODE solvers reach performance comparable to Julia's
> DifferentialEquations.jl while retaining Rust's substantially lower memory
> usage observed in earlier `orskit` experiments?

The working goals are:

- port representative high-value OrdinaryDiffEq.jl algorithms faithfully,
  beginning with `Tsit5` for non-stiff systems;
- compare numerical results directly with Julia and use Julia solver outputs
  as the compliance oracle;
- measure runtime, allocations, and peak memory on equivalent workloads;
- design the hot path around reusable caller-owned or solver-owned buffers;
- preserve a small, idiomatic Rust API without hiding allocation costs;
- record discrepancies instead of silently weakening tolerances.

The solver target is native OrdinaryDiffEq.jl **ODE algorithm parity**.
SDEs, RODEs, DDEs, DAEs, boundary-value problems, and external solver wrappers
are explicitly out of scope. Basic discrete and continuous event callbacks and
`save_at` sampling are implemented. Tsit5 and DPRKN6 have method-specific
high-order interpolation, with retained post-solve Tsit5 segments; dense
extensions for the other method families, arbitrary numeric types, and
sensitivities remain separate API features and are not implied by algorithm
parity.

## Method

Each algorithm is developed in three layers:

1. implement and unit-test the numerical kernel in Rust;
2. generate reference results with the corresponding Julia algorithm and
   compare endpoints, saved trajectories, and solver statistics;
3. benchmark matched Rust and Julia workloads for elapsed time and peak memory.

Julia tests use an isolated project under `tests/julia`. Rust tests remain
usable without Julia; cross-language compliance tests are explicit so normal
`cargo test` runs stay fast and deterministic.

Core problem, solution, and driver types are exported at the crate root.
Concrete solvers live under a family and implementation module:

```rust
use differential_equations::solvers::explicit::tsit5::Tsit5;
use differential_equations::{OdeProblem, SolveOptions, solve};
```

Family façades such as `solvers::explicit::Tsit5` provide shorter focused
imports. The historical `differential_equations::algorithms` namespace remains
an alias for `solvers`, including its glob-import prelude.

Stiff problems may optionally provide an analytic state Jacobian. Implicit and
Rosenbrock methods use it directly; otherwise they fall back to finite
differences. The callback must fill a row-major `dimension × dimension`
matrix:

```rust
let problem = OdeProblem::new(rhs, initial_state, time_span, parameters)
    .with_jacobian(|jacobian, state, parameters, time| {
        // jacobian[i * state.len() + j] = ∂fᵢ/∂uⱼ
    });
```

Discrete and scalar zero-crossing callbacks can change the state, continue the
solve, or terminate it. Solver caches are invalidated automatically after an
effect:

```rust
use differential_equations::{CallbackAction, OdeProblem};

let problem = OdeProblem::new(rhs, initial_state, time_span, parameters)
    .with_continuous_callback(
        |state, _parameters, _time| state[0],
        |state, _parameters, _time| {
            state[0] = -state[0];
            CallbackAction::Continue
        },
    );
```

Set `SolveOptions::save_at` to ordered output times. Like SciML's `saveat`, a
non-empty list replaces ordinary `SaveMode` output. The precise feature matrix
and known callback/interpolation limits are in
[`docs/FEATURE_COVERAGE.md`](docs/FEATURE_COVERAGE.md).

Tableau-defined explicit Runge–Kutta methods share one generic kernel. The
named algorithms (`Rk4`, `Tsit5`, `Dp5`, and others) are zero-sized facades over
`ExplicitRungeKutta<T>`. New methods can provide their coefficients by
implementing `ButcherTableau`; malformed dimensions, non-finite coefficients,
and invalid FSAL layouts are rejected before integration. Solver workspaces use
flat stage-major storage with separate candidate, error, and temporary arrays
so component loops remain contiguous and SIMD-friendly.

For file-based extension, `define_explicit_rk_from_file!` turns a TOML tableau
resource into a validated zero-sized algorithm at compile time. There is no
runtime parser or dynamic dispatch. See
[`docs/TABLEAU_RESOURCES.md`](docs/TABLEAU_RESOURCES.md) for the schema and a
complete downstream example.

Run both test layers with:

```console
cargo test
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

If the pin check fails after cloning or changing Julia dependencies, run the
same `pinned_environment.jl` command without `--check` once to bind the full
OrdinaryDiffEq subpackage closure to the reference Git revision.

Run the matched 31-algorithm steady-state benchmark matrix with:

```powershell
./benchmarks/run.ps1 -Repetitions 20
```

If Julia is not on `PATH`, pass its executable explicitly with
`-JuliaPath <path-to-julia>`.

Raw Rust and Julia measurements plus a ratio table are written beneath
`benchmarks/results/`. Allocation totals exclude compilation and warm-up.
For the reproducible, VM-per-case speed/RSS/allocation harness, see
[`benchmarks/cloud/README.md`](benchmarks/cloud/README.md); it is designed to
run through `gcloud` and never starts cloud resources on its own.
The exact generated implemented/remaining algorithm inventory is maintained in
[`docs/ODE_PARITY_INVENTORY.md`](docs/ODE_PARITY_INVENTORY.md); the coverage
policy and interpretation are in
[`docs/ALGORITHM_COVERAGE.md`](docs/ALGORITHM_COVERAGE.md).

Independent parameter sweeps and ensembles can run sequentially or on Rayon's
global thread pool. Results always remain in input order, and each case keeps
its own success or failure:

```rust
use differential_equations::solvers::explicit::tsit5::Tsit5;
use differential_equations::{
    ExecutionPolicy, OdeProblem, SolveOptions, solve_ensemble,
};

let outcomes = solve_ensemble(
    [0.5, 1.0, 2.0],
    |initial| OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![initial],
        (0.0, 1.0),
        (),
    ),
    Tsit5,
    &SolveOptions::default(),
    ExecutionPolicy::Parallel,
);
assert!(outcomes.iter().all(|case| case.result.is_ok()));
```

## Roadmap

- [x] Define the ODE problem, solver options, solution, and statistics API.
- [x] Implement adaptive `Tsit5` with reusable stage storage.
- [x] Validate `Tsit5` against OrdinaryDiffEq.jl on scalar and vector problems.
- [x] Implement the shared explicit Runge–Kutta kernel plus Euler, midpoint,
      Heun, Ralston, RK4, BS3, DP5, RKM, Ralston4, and Alshina2/3.
- [x] Implement fixed-step Adams–Bashforth methods AB3/4/5 and
      Adams–Bashforth–Moulton methods ABM32/43/54.
- [x] Establish dense Newton/Jacobian/linear-solve infrastructure and fixed
      Implicit Euler, Implicit Midpoint, and Trapezoid methods.
- [x] Implement SSPRK22, SSPRK33, and adaptive SSPRK43.
- [x] Implement adaptive Rosenbrock23 with one reused LU factorization per step.
- [x] Add a reproducible pinned-upstream inventory and exact-commit Julia
      compliance environment.
- [x] Expand to variable Adams, Verner, low-storage/SSP RK, TRBDF2,
      Rosenbrock/Rodas, and initial second-order symplectic families.
- [x] Port every in-scope native OrdinaryDiffEq.jl ODE algorithm family and
      verify all 345 included public names against the pinned Julia revision.
- [x] Establish matched runtime/allocation benchmarks and a reproducible
      peak-RSS cloud harness.
- [x] Add Rayon-backed APIs for parallel independent solve and ensemble cases.
- [x] Add compile-time TOML tableau resources for zero-overhead downstream
      explicit Runge--Kutta method definitions.
- [x] Add basic discrete/continuous callbacks and save-at behavior.
- [x] Complete the dense-output lifecycle: pinned method-specific extensions
      where upstream provides them, honest Hermite/partitioned fallbacks
      elsewhere, continuous-root interpolation, and retained post-solve
      segments.
- [x] Select `Rodas5P` as the default stiff solver for the current regular-ODE
      scope from the matched stiff-candidate benchmark slice.

The unattended implementation runbook is in
[`docs/OVERNIGHT_EXECUTION_PLAN.md`](docs/OVERNIGHT_EXECUTION_PLAN.md). Agent
delegation rules are in
[`docs/AGENT_OPERATING_RULES.md`](docs/AGENT_OPERATING_RULES.md), and the
copy/paste runner prompt is in
[`docs/OVERNIGHT_RUN_PROMPT.md`](docs/OVERNIGHT_RUN_PROMPT.md).

## Status

Beta. Implemented algorithms now use method-specific kernels and are covered by
Rust regression tests, with a growing set of comparisons against the pinned
Julia SciML reference environment. The API may still change before 1.0, and
users should validate solver choice and tolerances for scientific or production
workloads.
