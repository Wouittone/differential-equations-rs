//! Beta Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! The crate is in beta. Its API may still change while numerical compliance,
//! performance, and memory behavior are established.

#![forbid(unsafe_code)]

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

pub use callback::{CallbackAction, EventDirection};
pub use differential_equations_tableau_macros::define_explicit_rk_from_file;
pub use ensemble::{
    CaseOutcome, ExecutionPolicy, solve_batch, solve_batch_sequential, solve_ensemble,
    solve_ensemble_sequential,
};
#[cfg(feature = "parallel")]
pub use ensemble::{solve_batch_parallel, solve_ensemble_parallel};
pub use error::ConfigurationError;
pub use event::DEFAULT_EVENT_TOLERANCE;
pub use operator_problem::{LieGroupProblem, LinearOperatorProblem};
pub use problem::{MassMatrixOdeProblem, OdeProblem, SplitOdeProblem};
pub use semilinear::{SemilinearOdeProblem, solve_exponential};
pub use solution::{InterpolationError, Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use solvers as algorithms;
pub use solvers::explicit::split_euler::{SplitOdeAlgorithm, solve_split, solve_split_euler};
pub use solvers::second_order::general::{
    SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolution, SecondOrderSolveError,
    solve_second_order,
};
pub use solvers::second_order::symplectic::{
    SymplecticAlgorithm, SymplecticSolution, SymplecticSolveError, SymplecticTableau,
    solve_symplectic,
};
