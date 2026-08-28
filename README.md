# differential-equations-rs

A beta Rust port of the ODE solvers in Julia's
[DifferentialEquations.jl ecosystem](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).

The minimum supported Rust version (MSRV) is 1.85.

## Quickstart

The crate is still protected by an intentional publication lock while its
prerelease API is finalized. From a checkout, add it as a path dependency:

```toml
[dependencies]
differential-equations = { path = "../differential-equations-rs" }
```

After publication, replace `path` with the released version. Define an
in-place right-hand side, select an algorithm from its family, and call
`solve`:

```rust
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], rate: &f64, _time: f64| {
            derivative[0] = rate * state[0];
        },
        [1.0],
        (0.0, 1.0),
        -2.0,
    );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);

    let solution = solve(&problem, Tsit5, &options)?;
    println!("u(1) = {}", solution.last_state()[0]);
    println!("accepted steps = {}", solution.stats().accepted_steps);
    Ok(())
}
```

Run the packaged version as `cargo run --example quickstart`. The state is a
contiguous `f64` vector; every right-hand-side call must overwrite the complete
derivative slice.

## Choosing a solver

Start with the simplest family that matches the problem:

| Problem | Suggested starting point | Canonical import |
| --- | --- | --- |
| Non-stiff, adaptive first-order ODE | Tsit5 | `solvers::explicit::Tsit5` |
| Stiff first-order ODE | Rodas5P | `solvers::rosenbrock::Rodas5P` |
| Known fixed step, simple baseline | RK4 | `solvers::explicit::Rk4` |
| Automatic non-stiff/stiff fallback | DefaultOdeAlgorithm | `solvers::automatic::DefaultOdeAlgorithm` |
| Separable second-order or Hamiltonian system | RKN or symplectic family | `solvers::second_order` |
| Independent parameter sweep | `solve_ensemble` | crate root |

Stiff solvers benefit from an analytic Jacobian supplied with
`OdeProblem::with_jacobian`; otherwise they use finite differences. Fixed-step
algorithms require `adaptive = false` and a positive initial step. Solver names
and numerical coverage are detailed in the
[algorithm coverage guide](docs/ALGORITHM_COVERAGE.md), while callbacks, dense
output, and problem-type limits are in the
[feature coverage guide](docs/FEATURE_COVERAGE.md).

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `parallel` | yes | Enables Rayon-backed batch and ensemble entry points. |
| `allocation-metrics` | no | Enables allocation instrumentation used by benchmark targets. |

Disable default features when the application does not need Rayon:

```toml
differential-equations = { version = "0.1.0-beta.1", default-features = false }
```

## Migrating prerelease imports

The supported solver surface is hierarchical. Core problem, option, solution,
and primary driver types remain at the crate root; algorithms and specialized
drivers live below `solvers`. Replace prerelease compatibility paths as
follows:

| Previous path or pattern | Canonical replacement |
| --- | --- |
| `differential_equations::algorithms::<Name>` | `differential_equations::solvers::<family>::<Name>` |
| `differential_equations::solvers::*` | Import the required `solvers::<family>` items explicitly |
| `algorithms::explicit::low_storage::*` | `solvers::explicit::low_storage_rk::*` |
| `algorithms::implicit::basic::*` | `solvers::implicit::general::*` |
| `algorithms::implicit::diagonally_implicit::*` | `solvers::implicit::sdirk::*` |
| `algorithms::implicit::fully_implicit::*` | `solvers::implicit::firk::*` |
| `algorithms::second_order::structural::*` | `solvers::second_order::general::*` |
| `algorithms::interaction_picture::*` | `solvers::exponential::rkip::*` |
| `algorithms::amf::*` | `solvers::rosenbrock::amf::*` |
| Root split-ODE traits and drivers | `solvers::explicit::split_euler::*` |
| Root second-order traits and drivers | `solvers::second_order::*` |
| Root symplectic traits and drivers | `solvers::second_order::symplectic::*` |

Family-level solver facades such as `solvers::explicit::Tsit5` are preferred
for ordinary use. Implementation-module paths are public for users who need a
specific kernel or extension type. The exact changes for the pending release
are also recorded in the [changelog](CHANGELOG.md).

## Goals

This project began as a proof of concept around a specific performance
question:

> Can Rust ODE solvers reach performance comparable to Julia's
> DifferentialEquations.jl while retaining Rust's substantially lower memory
> usage observed in earlier `orskit` experiments?

The project now uses that comparison as one release gate alongside numerical
compliance, API stability, and predictable memory behavior. Its working goals
are:

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
`save_at` sampling are implemented. Several explicit, Rosenbrock, SSP, and
second-order methods have method-specific continuous extensions; other
implemented methods use their documented retained fallback. Arbitrary numeric
types and sensitivities remain separate API features and are not implied by
algorithm parity.

## Method

Each algorithm is developed in three layers:

1. implement and unit-test the numerical kernel in Rust;
2. generate reference results with the corresponding Julia algorithm and
   compare endpoints, saved trajectories, and solver statistics;
3. benchmark matched Rust and Julia workloads for elapsed time and peak memory.

In a source checkout, Julia tests use an isolated project under `tests/julia`
and the pinned `reference/OrdinaryDiffEq.jl` Git submodule. Rust tests remain
usable without Julia; cross-language compliance tests are explicit so normal
`cargo test` runs stay fast and deterministic.

Core problem, solution, and driver types are exported at the crate root.
Concrete solvers live under a family and implementation module:

```rust
use differential_equations::solvers::explicit::tsit5::Tsit5;
use differential_equations::{OdeProblem, SolveOptions, solve};
```

Family façades such as `solvers::explicit::Tsit5` provide shorter focused
imports. Prefer these canonical family paths in downstream code; implementation
modules remain useful when distinguishing closely related solver variants.

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
resource-backed low-order algorithms, including `Rk4`, are zero-sized facades
over `ExplicitRungeKutta<T>`; methods with specialized staging or dense output
retain their dedicated kernels. New methods can provide their coefficients by
implementing `ButcherTableau`; malformed dimensions, non-finite coefficients,
and invalid FSAL layouts are rejected before integration. Solver workspaces use
flat stage-major storage with separate candidate, error, and temporary arrays
so component loops remain contiguous and SIMD-friendly.

For file-based extension, `define_explicit_rk_from_file!` turns a TOML tableau
resource into a validated zero-sized algorithm at compile time. There is no
runtime parser or dynamic dispatch. See
[`docs/TABLEAU_RESOURCES.md`](docs/TABLEAU_RESOURCES.md) for the schema and a
complete downstream example.

From a source checkout, run both test layers with:

```console
git submodule update --init --recursive
cargo test
julia --project=tests/julia tests/julia/pinned_environment.jl --check
julia --project=tests/julia tests/julia/runtests.jl
```

If the pin check fails after cloning or changing Julia dependencies, run the
same `pinned_environment.jl` command without `--check` once to bind the full
OrdinaryDiffEq subpackage closure to the checked-out reference submodule. The
Julia manifest uses local submodule paths and does not fetch OrdinaryDiffEq
from a second remote checkout.

Run the matched 31-algorithm steady-state benchmark matrix with:

```powershell
./benchmarks/run.ps1 -Repetitions 20
```

If Julia is not on `PATH`, pass its executable explicitly with
`-JuliaPath <path-to-julia>`.

Raw Rust and Julia measurements plus a ratio table are written beneath
`benchmarks/results/`. Allocation totals exclude compilation and warm-up.
For lightweight regression tracking on every pull request, run the
Criterion-compatible CodSpeed suite:

```console
cargo bench --bench solver_performance
```

Its stable benchmark IDs cover representative explicit, stiff, dense-output,
and sequential/parallel ensemble paths. Baseline comparison and CI details are
in the [performance regression guide](docs/BENCHMARKING.md).

For the reproducible, VM-per-case speed/RSS/allocation harness, see the
[cloud benchmark guide](https://github.com/Wouittone/differential-equations-rs/blob/main/benchmarks/cloud/README.md);
it is designed to run through `gcloud` and never starts cloud resources on its
own.

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

## Status

Prerelease. The pinned native-ODE algorithm-name scope is implemented and has
matched Rust/Julia compliance coverage. The API may still change before 1.0,
and users should validate solver choice and tolerances for scientific or
production workloads. The intentional Cargo publication lock is removed only
as part of the reviewed release process.

## Project documentation

- [Feature coverage and limitations](docs/FEATURE_COVERAGE.md)
- [Algorithm coverage](docs/ALGORITHM_COVERAGE.md)
- [Performance regression benchmarks](docs/BENCHMARKING.md)
- [Compile-time tableau resources](docs/TABLEAU_RESOURCES.md)
- [Pinned upstream scope](docs/UPSTREAM_SCOPE.md)
- [Changelog and migration notes](CHANGELOG.md)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Release process](docs/RELEASING.md)

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

Portions are derived from MIT-licensed SciML/OrdinaryDiffEq.jl. Its retained
copyright and permission notice are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion in this project are licensed under the same dual-license terms.
