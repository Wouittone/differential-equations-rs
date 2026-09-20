//! Numerical ordinary differential equation solvers with a compact Rust API.
//!
//! The crate provides in-place first-order ODE problems, solver options,
//! callbacks (including vector event conditions and exact preset-time effects),
//! retained dense output, and
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
//!   [`CallbackSet`] composes callback policies before attaching them to a
//!   first-order or split problem and provides initialization/finalization
//!   hooks around successful integration. Vector continuous callbacks group
//!   several root functions; they are separate from ndarray state shape.
//!   [`callbacks::DomainGuard`] rejects out-of-domain candidate states before
//!   callback effects or trajectory saving.
//!   [`callbacks::PositiveDomain`] cheaply restricts upcoming steps and clamps
//!   accepted states to preserve componentwise non-negativity.
//!   [`callbacks::ManifoldProjection`] enforces implicit conservation laws
//!   with rectangular residuals and typed nonlinear failure.
//!   [`callbacks::GeneralDomain`] couples that projection engine with
//!   predictive step control for a user-defined domain residual.
//!   [`CallbackAction::ContinueWithStepSize`] overrides the next proposed step;
//!   Rust interior-mutability types such as [`std::cell::Cell`] let sequential
//!   callback effects update parameters without imposing mutable problem
//!   ownership on every solve.
//! - [`callbacks`] contains reusable integration-time policies. Its periodic
//!   and iterative schedulers use constant memory; [`callbacks::IterativeCallback`]
//!   chooses each next event from the state after the previous effect.
//!   [`callbacks::TerminateSteadyState`] stops when the problem's derivatives
//!   satisfy independent componentwise tolerances.
//!   Its function-calling policy marks
//!   read-only observations explicitly, and its step-size limiter applies
//!   dynamic stability bounds without interfering with smaller adaptive steps.
//! - [`SolveOptions`] controls tolerances, step sizes, exact time stops, saved
//!   output, and dense output retention; [`solve`] runs an [`OdeAlgorithm`].
//! - [`solvers::explicit`] is a good starting point for non-stiff problems,
//!   [`solvers::rosenbrock`] contains linearly implicit stiff methods, and
//!   [`solvers::automatic`] contains in-flight non-stiff/stiff composites with
//!   configurable hysteresis and observable branch statistics.
//! - [`tableau`] exposes the extension surface for defining explicit
//!   Runge--Kutta and Runge--Kutta--Nyström methods from compile-time-validated
//!   JSON resources, plus typed data for specialized solver families.
//! - [`OdeProblem::builder`] names the initial state, time span, parameters,
//!   and ndarray evaluation style. [`OdeProblem::from_array`] remains the
//!   compact constructor for scalar, vector, and matrix states while numerical
//!   kernels retain contiguous flat workspaces.
//!   [`OdeProblem::from_array_out_of_place`] accepts functions returning arrays;
//!   [`OdeFunction`] unifies them with in-place closures and propagates errors.
//!   [`solvers::second_order::SecondOrderOdeProblem`] provides matching ndarray
//!   constructors while keeping velocity and position partitions separate.
//! - [`solve_ensemble`] and [`solve_batch`] preserve input order. The default
//!   `parallel` feature also enables their Rayon-backed variants.
//!
//! # Features
//!
//! The public Cargo feature surface is intentionally small:
//!
//! - `parallel` is enabled by default and adds Rayon-backed independent solves.
//! - `allocation-metrics` enables repository benchmark instrumentation and is
//!   not needed by ordinary users.
//!
//! Disabling default features selects sequential ensemble execution. State
//! shape is part of the core API, not a feature: ndarray scalar, vector, and
//! matrix adapters remain available with `default-features = false`. There are
//! no `vector`, `matrix`, or `ndarray` feature flags.
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
pub mod callbacks;
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
pub mod state_layout;
pub mod stepping;
pub mod solvers;
pub mod tableau;
pub mod tolerances;

pub use callback::{CallbackAction, CallbackSave, CallbackSet, EventCrossing, EventDirection};
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
pub use problem::{MutableFunction, OdeFunction, OdeProblem, OdeProblemBuilder, SplitOdeProblem};
pub use semilinear::SemilinearOdeProblem;
pub use solution::{DenseSegmentData, PortableDenseSegment, SolutionData, InterpolationQuality,
    InterpolationError, Solution, SolutionConstructionError, SolverStats};
pub use solver::{
    AutomaticPairIncompatibility, OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve,
};
