# differential-equations-rs

Rust ordinary differential equation solvers inspired by Julia's
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). The crate is
currently a beta and requires Rust 1.85 or newer.

The supported API is intentionally hierarchical: problem, option, solution,
and driver types live at the crate root, while algorithms live under
`solvers::<family>`.

```rust
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

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

## Choosing an algorithm

| Problem | Suggested starting point | Import |
| --- | --- | --- |
| Non-stiff first-order ODE | Tsit5 | `solvers::explicit::Tsit5` |
| Stiff first-order ODE | Rodas5P | `solvers::rosenbrock::Rodas5P` |
| Simple fixed-step baseline | Rk4 | `solvers::explicit::Rk4` |
| Automatic non-stiff/stiff fallback | DefaultOdeAlgorithm | `solvers::automatic::DefaultOdeAlgorithm` |
| Separable second-order system | RKN or symplectic family | `solvers::second_order` |

Stiff solvers can use an analytic Jacobian supplied through
`OdeProblem::with_jacobian`; otherwise they use finite differences. Fixed-step
algorithms require adaptive stepping to be disabled and an initial step to be
provided.

## Features and current scope

- `parallel` is enabled by default and provides Rayon-backed batch and ensemble
  solves. Disable default features for a sequential-only dependency.
- `allocation-metrics` is development instrumentation used by the comparison
  benchmark; applications do not need it.
- First-order solvers keep contiguous `f64` workspaces. `OdeProblem::new`
  retains the original flat-vector API, while `OdeProblem::from_array` accepts
  ndarray scalars, vectors, and matrices and returns shape-aware solution
  views without changing the numerical kernels.
- The crate supports discrete, scalar continuous, vector continuous, and
  preset-time callbacks. Vector continuous callbacks group several event
  functions into one evaluation and report a signed crossing mask when one or
  more roots occur simultaneously; this is independent of whether the ODE
  state itself is a scalar, vector, or matrix. Preset
  callback times automatically become exact integration stops, including for
  split and second-order problems. `CallbackSave` controls whether the state
  before an effect, after it, both, or neither is added to the trajectory;
  continuous callbacks save both by default and discrete callbacks save the
  affected state. `CallbackSet` and `SecondOrderCallbackSet` let applications
  assemble ordered callback policies separately from their problems, run
  initialization hooks before initial conditions are tested, and synchronize
  finalized endpoint states after successful solves. Effects can return
  `CallbackAction::ContinueWithStepSize` to override the next adaptive or
  fixed-step proposal. Parameters remain ordinary Rust values: use `Cell` or
  `RefCell` when a sequential callback must mutate them, or a synchronized
  type when intentionally sharing them across threads. Mutating effects
  invalidate solver caches so subsequent right-hand-side and Jacobian calls
  observe the updated parameter value. Observation-only effects can return
  `CallbackAction::ContinueUnmodified` to retain those caches; that action must
  not be used after changing state or right-hand-side parameters through
  interior mutability. Saved-time sampling, retained dense output, and the
  documented solver families are also supported. SDEs, DDEs,
  boundary-value problems, and external solver wrappers are out of scope.

The main package remains protected by `publish = false` while its prerelease
API is finalized. Use it as a path or Git dependency until a release is
published.

## Scalar, vector, and matrix states

The ndarray entry point uses one API for zero-, one-, and two-dimensional
states. Its right-hand side receives views with the original shape; the adapter
to the solver's flat workspace is monomorphized and allocation-free during
integration.

```rust
use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, array};
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SolveOptions, solve};

let problem = OdeProblem::from_array(
    |mut du: ArrayViewMut2<'_, f64>, u: ArrayView2<'_, f64>, _: &(), _: f64| {
        du.zip_mut_with(&u, |du, u| *du = -*u);
    },
    array![[1.0, 2.0], [3.0, 4.0]],
    (0.0, 1.0),
    (),
);
let solution = solve(&problem, Tsit5, &SolveOptions::default())?;
assert_eq!(solution.last_state_array().shape(), &[2, 2]);
# Ok::<(), differential_equations::SolveError>(())
```

Use `arr0(value)` for a scalar, `array![...]` for a vector, and
`array![[...], [...]]` for a matrix. `initial_state_array`, `state_array`,
`last_state_array`, and `interpolate_array` retain ndarray dimensionality.
Flat slice access remains available when callers want the contiguous fast
path directly.

For equations that return a value, use `OdeProblem::from_array_out_of_place`
(or `SplitOdeProblem::from_array_out_of_place` for two component functions):

```rust
use differential_equations::ndarray::{ArrayView2, array};
use differential_equations::OdeProblem;

let problem = OdeProblem::from_array_out_of_place(
    |u: ArrayView2<'_, f64>, _: &(), _| -&u,
    array![[1.0, 2.0], [3.0, 4.0]],
    (0.0, 1.0),
    (),
);
```

The returned derivative must have the same shape as the state; a mismatch
returns `SolveError::DerivativeShapeMismatch`, including during Jacobian or
callback evaluations. Nonstandard array layouts are supported. Returning an
owned array can allocate on every evaluation, so the in-place entry point
remains useful for allocation-sensitive applications. Both forms use the
same numerical kernels and retain the same callback and dense-output APIs.

`OdeFunction<P>` is the common fallible evaluation interface. Existing
in-place closures implement it automatically. Custom algorithm implementations
should now bound their function type by `OdeFunction<P>` rather than `Fn(...)`;
custom functions can implement it to return a typed error. Direct calls to
`SplitOdeProblem::evaluate_explicit` and `evaluate_implicit` now return a
`Result` that callers must handle.

### Second-order array states

`SecondOrderOdeProblem::from_array` and `from_array_out_of_place` offer the
same ndarray forms for `q'' = f(v, q, p, t)`. Velocity and position remain
separate and must have exactly matching shapes; constructors return a
`Result` to report incompatible partitions. This applies to the RKN,
structural, and symplectic solvers.

```rust
use differential_equations::ndarray::{arr0, ArrayView0};
use differential_equations::solvers::second_order::SecondOrderOdeProblem;

let problem = SecondOrderOdeProblem::from_array_out_of_place(
    |_: ArrayView0<'_, f64>, q: ArrayView0<'_, f64>, _: &(), _| -&q,
    arr0(0.0), // Initial velocity.
    arr0(1.0), // Initial position.
    (0.0, 1.0),
    (),
)?;
# let _ = problem;
# Ok::<(), differential_equations::ConfigurationError>(())
```

Shape-aware `with_array_*_callback` methods receive velocity before position.
Solutions provide `position_array`, `velocity_array`, their `last_*` forms,
and `interpolate_array`. Interpolation preserves the existing tuple order:
`(velocity, position)` for `SecondOrderSolution`, `(position, velocity)` for
`SymplecticSolution`. Scalar arrays retain their zero-dimensional shape.

Custom second-order algorithms now bound their acceleration type by
`SecondOrderFunction<P>`. Existing in-place closures implement this trait
automatically; direct `evaluate_acceleration` calls now return a `Result`.
Returned acceleration shapes are checked at each evaluation. In-place
fixed-rank ndarray adapters do not allocate per evaluation, while returned
owned arrays can allocate. This remains the `q' = v` specialization, not yet
a general dynamical problem with an independent position-rate function.

## Reusable callback policies

The `callbacks` module provides common policies as independently composable
callback sets. `PeriodicCallback` schedules effects at exact integration times
without materializing every time in memory, supports phase offsets and forward
or backward solves, and can optionally affect the initial and final states.
`IterativeCallback` chooses each next absolute event time from the updated
state and parameters. It retains only one pending time, resets each solve,
and accepts `None` to stop scheduling. Non-finite or non-advancing choices
return a typed solve error. Initialization chooses the first time without
firing or saving an effect unless `with_initial_affect(true)` is selected.
`TerminateSteadyState` stops when every derivative satisfies
`abs(du[i]) <= max(absolute[i], relative[i] * abs(u[i]))`, using the problem's
own equations. Its independent default tolerances are `1e-8` and `1e-6`;
scalar or componentwise overrides and a minimum absolute time are supported.
The criterion follows the
[SciML steady-state policy](https://docs.sciml.ai/DiffEqCallbacks/stable/steady_state/).
It checks the total derivative for split problems and both acceleration and
velocity for second-order problems. Checks reuse scratch storage, preserve
solver caches when not terminating, and include their derivative evaluations
in the statistics. A small instantaneous derivative does not guarantee a
long-term equilibrium, especially for time-dependent equations.
`FunctionCallingCallback` observes the initial state, every accepted step, or
an explicit set of exact times without invalidating solver caches.
`StepsizeLimiter` applies a state-dependent stability bound, such as a CFL
limit, while leaving adaptive controllers free to choose a smaller step.
`DomainGuard` rejects a finite candidate state before callbacks or saving and
retries the attempt with a configurable reduction factor. It works with
ordinary, split, and partitioned second-order problems.
`PositiveDomain` instead checks a cheap forward extrapolation before an
attempt, repeatedly reduces an unsafe upcoming step, applies a `0.9` safety
factor, and clamps negative accepted components to zero. It defaults to the
solve's absolute tolerance, supports an explicit override, and works for
ordinary and split first-order problems with scalar, vector, or matrix state.
`ManifoldProjection` enforces one or more implicit conservation constraints
after initialization and every accepted step. It supports rectangular
residuals, finite-difference or analytic Jacobians, backtracked Newton
corrections, and typed non-convergence errors for ordinary and split
first-order problems. Because projection changes endpoints after dense output
is constructed, requested samples that must be projected should also be listed
as exact time stops.
`GeneralDomain` combines the same projection engine with a predictive domain
check. It evaluates residuals at a forward-Euler extrapolation and its future
time, shrinks an unsafe proposal, and applies a `0.9` safety margin. Its
predictor tolerance defaults to the solve's absolute tolerance, independently
of the projection tolerance. For a region, define a residual that is zero
inside and positive outside; prediction uses a signed comparison, while
projection always targets the zero set. Already-satisfied, locally inactive
constraints are omitted from the correction system. As with the
[SciML policy](https://docs.sciml.ai/DiffEqCallbacks/stable/step_control/#DiffEqCallbacks.GeneralDomain),
nonlinear projection guarantees proximity only, not strict inequality-domain
membership. Use `PositiveDomain` when exact componentwise non-negativity is
required.

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
    |derivative: &mut [f64], _: &[f64], _: &(), _| derivative.fill(0.0),
    [0.0],
    (0.0, 1.0),
    (),
)
.with_callback_set(callbacks);
# let _ = problem;
# Ok::<(), differential_equations::ConfigurationError>(())
```

`PeriodicCallback`, `IterativeCallback`, `FunctionCallingCallback`,
`StepsizeLimiter`, `TerminateSteadyState`, and `DomainGuard` also construct
partitioned callback sets for second-order problems through
`into_second_order_callback_set`. `DomainGuard` checks actual
candidate states and may repeat their computation; `PositiveDomain` predicts
the next state before the attempt. `GeneralDomain` extends that predictive
control to user-defined residuals and uses `ManifoldProjection` for endpoint
corrections. Both predictive policies support ordinary and split first-order
problems; they do not yet expose partitioned second-order callback sets.

## Tableau extensions

Runge--Kutta data is stored as compile-time JSON resources below
`src/tableau/resources`. Resources are validated while compiling and parsed
lazily when their method is first used. Downstream crates can define an
explicit Runge--Kutta algorithm from their own JSON file with
`define_explicit_rk_from_file!`, or a drift/kick composition with
`tableau::define_symplectic_from_file!`. Embedded Runge--Kutta pairs may specify
`b_hat` instead of precomputed error weights; RKIP uses this same resource format.
All named symplectic compositions now
use individual resources and expose fallible `Method::tableau()` access with
`a()`/`b()` coefficient slices. Some other specialized families still retain
legacy embedded coefficient data; see
[the tableau resource guide](docs/TABLEAU_RESOURCES.md).

## Development

The ordinary Rust quality gates are:

```console
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets --all-features
cargo test --locked --workspace --all-targets --no-default-features
```

Cross-language compliance tests use the pinned `reference/OrdinaryDiffEq.jl`
submodule and the Julia project in `tests/julia`; normal Cargo tests do not
require Julia. Performance regression and matched Rust/Julia commands are in
[the benchmarking guide](docs/BENCHMARKING.md). Package and publication checks
are described in [the release guide](docs/RELEASING.md).

## License

Licensed under either the Apache License, Version 2.0 or the MIT License, at
your option.
