# differential-equations-rs

Fast, composable ordinary differential equation solvers for Rust, inspired by
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

[![CI](https://github.com/Wouittone/differential-equations-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Wouittone/differential-equations-rs/actions/workflows/ci.yml)
![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)
![MIT or Apache-2.0](https://img.shields.io/badge/license-MIT%20or%20Apache--2.0-blue.svg)

- Adaptive and fixed-step methods for non-stiff, stiff, split, and second-order problems
- One API for scalar, vector, and matrix states through `ndarray`
- Events, domain guards, manifold projection, dense output, and exact-time sampling
- Parallel ensemble solves with Rayon, enabled by default
- Extensible, compile-time validated JSON tableaus that load only when used

The Cargo package is `differential-equations-rs`; the Rust library name is the
shorter `differential_equations`.

## Install

After the 1.0 release is published:

```toml
[dependencies]
differential-equations-rs = "1.0"
```

Until then, use a checkout:

```toml
[dependencies]
differential-equations-rs = { path = "../differential-equations-rs" }
```

Version 1.0 requires Rust 1.85 or newer.

## Quick start

Solve `u' = -2u` from `t = 0` to `t = 1` with the adaptive Tsitouras 5/4
method:

```rust
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{solve, OdeProblem, SaveMode, SolveOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], rate: &f64, _time: f64| {
            du[0] = rate * u[0];
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
    Ok(())
}
```

Algorithms live under `solvers::<family>`; problem, option, solution, and
driver types stay at the crate root.

## Pick a solver

| Your problem | Start with | Import |
| --- | --- | --- |
| Non-stiff first-order ODE | `Tsit5` | `solvers::explicit::Tsit5` |
| Stiff first-order ODE | `Rodas5P` | `solvers::rosenbrock::Rodas5P` |
| Unknown or changing stiffness | `AutoTsit5<Rodas5P>` | `solvers::automatic::AutoTsit5` |
| Simple fixed-step baseline | `Rk4` | `solvers::explicit::Rk4` |
| Separable second-order system | RKN or symplectic method | `solvers::second_order` |

Stiff methods can use an analytic Jacobian supplied with
`OdeProblem::with_jacobian`; otherwise they use finite differences. Fixed-step
methods require `adaptive = false` and an initial step size.

Automatic methods switch kernels without restarting the integration. Saved
output, callbacks, mutable parameters, exact stops, dense interpolation, step
budgets, and statistics all remain continuous across a switch. The decision
history is visible through `SolverStats`.

## State shapes

For shaped states, prefer `OdeProblem::builder()`. It names every part of the
problem and makes the in-place versus out-of-place evaluation choice explicit.
The right-hand side sees the original ndarray shape while the solver keeps one
contiguous workspace internally.

```rust
use differential_equations::ndarray::{array, ArrayView2, ArrayViewMut2};
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{solve, OdeProblem, SolveOptions};

let problem = OdeProblem::builder()
    .initial_state(array![[1.0, 2.0], [3.0, 4.0]])
    .time_span((0.0, 1.0))
    .parameters(())
    .build_with_in_place_rhs(
        |mut du: ArrayViewMut2<'_, f64>,
         u: ArrayView2<'_, f64>,
         _: &(),
         _: f64| {
            du.zip_mut_with(&u, |du, u| *du = -*u);
        },
    );

let solution = solve(&problem, Tsit5, &SolveOptions::default())?;
assert_eq!(solution.last_state_array().shape(), &[2, 2]);
# Ok::<(), differential_equations::SolveError>(())
```

Use `arr0(value)` for a scalar, `array![...]` for a vector, and
`array![[...], [...]]` for a matrix. Shape-aware initial-state, solution, and
interpolation views retain that dimensionality. Flat slice access remains
available for callers that want it.

Prefer `build_with_in_place_rhs` when allocations matter. Use
`build_with_out_of_place_rhs` when returning an owned derivative is more
natural. Returned shapes are checked and mismatches produce a typed error.
The positional `from_array` and `from_array_out_of_place` constructors remain
available for compact or existing code.

Second-order problems offer the same forms through
`SecondOrderOdeProblem::from_array` and `from_array_out_of_place`. Velocity and
position stay separate, preserve their shape, and must have matching
dimensions.

## Events and constraints

Callbacks compose independently from the problem, so event policy can be
reused across solvers and state shapes.

| Callback | Use it for |
| --- | --- |
| `PeriodicCallback` | Effects at a fixed interval without storing every event time |
| `IterativeCallback` | Choosing the next event time from the updated state |
| `FunctionCallingCallback` | Observing accepted states without invalidating caches |
| `StepsizeLimiter` | CFL or other state-dependent stability limits |
| `TerminateSteadyState` | Stopping when the derivative is sufficiently small |
| `DomainGuard` | Rejecting invalid candidate states and retrying with a smaller step |
| `PositiveDomain` | Predicting and preserving componentwise non-negativity |
| `ManifoldProjection` | Enforcing implicit conservation constraints |
| `GeneralDomain` | Predictive domain control plus nonlinear projection |

Discrete, scalar continuous, vector continuous, and preset-time callbacks are
supported. Effects can continue, terminate, or replace the next step-size
proposal. Callback save policy controls whether the trajectory records the
state before an effect, after it, both, or neither.

```rust
use differential_equations::callbacks::PeriodicCallback;
use differential_equations::{CallbackAction, OdeProblem};

let callbacks = PeriodicCallback::new(0.1)
    .with_final_affect(true)
    .into_callback_set((0.0, 1.0), |state, _: &(), _| {
        state[0] += 1.0;
        CallbackAction::Continue
    })?;

let problem = OdeProblem::new(
    |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
    [0.0],
    (0.0, 1.0),
    (),
)
.with_callback_set(callbacks);
# let _ = problem;
# Ok::<(), differential_equations::ConfigurationError>(())
```

Most general callbacks also have partitioned forms for second-order problems.
`PositiveDomain`, `ManifoldProjection`, and `GeneralDomain` currently target
ordinary and split first-order problems.

## Parallel ensembles

The default `parallel` feature uses Rayon for independent batch and ensemble
solves. Disable default features for sequential execution:

```toml
[dependencies]
differential-equations-rs = { version = "1.0", default-features = false }
```

State shapes are not feature-gated. Scalar, vector, and matrix adapters remain
available in sequential builds.

| Feature | Default | Purpose |
| --- | :---: | --- |
| `parallel` | Yes | Rayon-backed batch and ensemble solves |
| `allocation-metrics` | No | Benchmark instrumentation for this repository |

## Add your own method

Solver coefficients live in canonical JSON resources, not generated Rust
files. Every resource is validated at compile time, embedded with the crate,
and parsed lazily the first time its method is used. Unused tableaus therefore
consume no runtime initialization or retained memory.

Downstream crates can define algorithms from their own JSON files:

- `define_explicit_rk_from_file!`
- `define_rkn_from_file!`
- `define_low_storage_rk_from_file!`
- `define_symplectic_from_file!`

The [tableau resource guide](docs/TABLEAU_RESOURCES.md) documents each format,
validation rule, and extension macro. A complete starting point is available
in [the file-based tableau example](examples/tableau_from_file.rs).

## Scope

Included today:

- Ordinary and split first-order ODEs
- RKN, structural, and symplectic second-order systems
- Adaptive and fixed-step integration
- Dense output and requested-time sampling
- Exact stops, reusable callback sets, and solver statistics

SDEs, DDEs, boundary-value problems, and wrappers around external solvers are
outside the 1.0 scope.

## Development and validation

```console
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets --all-features
cargo test --locked --workspace --all-targets --no-default-features
```

The test suite includes scalar, vector, and matrix problems. Cross-language
certification uses the pinned `reference/OrdinaryDiffEq.jl` submodule and the
Julia project in `tests/julia`; ordinary Cargo tests do not require Julia.

- [Benchmarking and Rust/Julia comparisons](docs/BENCHMARKING.md)
- [Release policy and 1.0 checklist](docs/RELEASING.md)
- [Tableau resource format](docs/TABLEAU_RESOURCES.md)

API documentation is built with every feature enabled. CI treats missing docs,
broken intra-doc links, MSRV regressions, dependency-policy violations, and
changes to the frozen public API as failures.

## License

Licensed under either the Apache License, Version 2.0 or the MIT License, at
your option.
