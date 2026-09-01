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
  type when intentionally sharing them across threads. Every callback effect
  invalidates solver caches, so subsequent right-hand-side and Jacobian calls
  observe the updated parameter value.
  Saved-time sampling, retained dense output, and the
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

## Tableau extensions

Runge--Kutta data is stored as compile-time JSON resources below
`src/tableau/resources`. Resources are validated while compiling and parsed
lazily when their method is first used. Downstream crates can define an
explicit Runge--Kutta algorithm from their own JSON file with
`define_explicit_rk_from_file!`; see
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
