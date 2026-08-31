//! Numerical ordinary differential equation solvers with a compact Rust API.
//!
//! The crate provides in-place first-order ODE problems, solver options,
//! callbacks (including exact preset-time effects), retained dense output, and
//! ordered ensemble execution. Concrete
//! algorithms are grouped by family below [`solvers`]; core problem, solution,
//! and driver types remain at the crate root.
//!
//! # Quickstart
//!
//! ```
//! use differential_equations::solvers::explicit::Tsit5;
//! use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
//!
//! let problem = OdeProblem::new(
//!     |derivative: &mut [f64], state: &[f64], rate: &f64, _time: f64| {
//!         derivative[0] = rate * state[0];
//!     },
//!     [1.0],
//!     (0.0, 1.0),
//!     -2.0,
//! );
//! let options = SolveOptions::new()
//!     .with_tolerances(1.0e-9, 1.0e-9)
//!     .with_save(SaveMode::Endpoints);
//!
//! let solution = solve(&problem, Tsit5, &options)?;
//! assert!(solution.last_state()[0] < 0.14);
//! # Ok::<(), differential_equations::SolveError>(())
//! ```
//!
//! # API organization
//!
//! - [`OdeProblem`] describes a first-order initial-value problem and optional
//!   Jacobian or callback behavior. [`OdeProblem::with_preset_time_callback`]
//!   schedules effects at exact integration times without duplicating them in
//!   [`SolveOptions::time_stops`], while [`CallbackSave`] selects whether the
//!   callback's left limit, affected state, both, or neither are retained.
//! - [`SolveOptions`] controls tolerances, step sizes, exact time stops, saved
//!   output, and dense output retention; [`solve`] runs an [`OdeAlgorithm`].
//! - [`solvers::explicit`] is a good starting point for non-stiff problems,
//!   [`solvers::rosenbrock`] contains linearly implicit stiff methods, and
//!   [`solvers::automatic`] contains composite choices.
//! - [`tableau`] exposes the stable extension surface for defining explicit
//!   Runge--Kutta methods from compile-time-validated JSON resources.
//! - [`OdeProblem::from_array`] accepts ndarray scalar, vector, and matrix
//!   states while numerical kernels retain contiguous flat workspaces.
//! - [`solve_ensemble`] and [`solve_batch`] preserve input order. The default
//!   `parallel` feature also enables their Rayon-backed variants.
//!
//! # Features
//!
//! `parallel` is enabled by default and adds Rayon-backed independent solves.
//! Disable default features for a sequential-only dependency. The optional
//! `allocation-metrics` feature enables instrumentation used by repository
//! benchmark targets; it is not needed by ordinary users.
//!
//! # Errors and panics
//!
//! Public solve functions return [`SolveError`] for invalid dimensions,
//! non-finite inputs, unsupported option/algorithm combinations, and numerical
//! failure. Problem right-hand sides, Jacobians, and callbacks are user code;
//! a panic in one of them is not caught by the solver. In-place right-hand
//! sides and Jacobians must fill their complete output buffers, and callback
//! effects must preserve the problem's state dimension.
//!
//! # Further reading
//!
//! The packaged repository documentation includes the solver-selection and
//! scope guide in the [README] and the [compile-time tableau resource guide].
//! Contributors can use the
//! [performance regression guide] to compare stable benchmark IDs locally.
//!
//! [README]: https://github.com/Wouittone/differential-equations-rs/blob/main/README.md
//! [compile-time tableau resource guide]: https://github.com/Wouittone/differential-equations-rs/blob/main/docs/TABLEAU_RESOURCES.md
//! [performance regression guide]: https://github.com/Wouittone/differential-equations-rs/blob/main/docs/BENCHMARKING.md

#![forbid(unsafe_code)]
#![warn(missing_docs, rustdoc::broken_intra_doc_links)]

mod callback;
mod ensemble;
mod error;
mod event;
mod integrator;
mod linear;
mod operator_problem;
mod problem;
mod semilinear;
mod solution;
mod solver;
pub mod solvers;
pub mod tableau;

pub use callback::{CallbackAction, CallbackSave, EventDirection};
pub use ensemble::{
    CaseOutcome, ExecutionPolicy, solve_batch, solve_batch_sequential, solve_ensemble,
    solve_ensemble_sequential,
};
#[cfg(feature = "parallel")]
pub use ensemble::{solve_batch_parallel, solve_ensemble_parallel};
pub use error::ConfigurationError;
pub use event::DEFAULT_EVENT_TOLERANCE;
/// The ndarray version used by shape-aware ODE states.
pub use ndarray;
pub use operator_problem::{LieGroupProblem, LinearOperatorProblem};
pub use problem::{OdeProblem, SplitOdeProblem};
pub use semilinear::SemilinearOdeProblem;
pub use solution::{InterpolationError, Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use tableau::define_explicit_rk_from_file;
