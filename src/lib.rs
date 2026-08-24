//! Beta Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! The crate is in beta. Its API may still change while numerical compliance,
//! performance, and memory behavior are established.

#![forbid(unsafe_code)]

mod abdf2;
mod adams;
mod anas5;
mod autodp5;
mod bdf;
mod callback;
mod coefficients;
mod compatibility;
mod composites;
mod ensemble;
mod explicit_rk;
mod exponential_rk;
mod frk65;
mod generated_coefficients;
mod high_order;
mod implicit;
mod integrator;
mod irkn_coefficients;
mod linear;
mod low_storage_rk;
mod mebdf2;
mod pdirk;
mod prk;
mod problem;
mod qndf1;
mod qndf2;
mod qprk;
mod rkn_adaptive_coefficients;
mod rosenbrock;
mod rosenbrock_extended;
mod sdirk;
mod sdirk_cash4;
mod second_order;
mod semilinear;
mod solution;
mod solver;
mod split_euler;
mod ssprk_extended;
mod ssprk_kyk2014;
mod ssprk_kyk42;
mod ssprk_msvs;
mod stabilized;
mod stabilized_coefficients;
mod symplectic;
mod trbdf2;
mod tsit5;
mod variable_adams;
mod verner;

pub use callback::{CallbackAction, EventDirection};
pub use compatibility::algorithms;
pub use ensemble::{CaseOutcome, ExecutionPolicy, solve_batch, solve_ensemble};
pub use exponential_rk::ExponentialAlgorithm;
pub use problem::{MassMatrixOdeProblem, OdeProblem, SplitOdeProblem};
pub use second_order::{
    SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolution, SecondOrderSolveError,
    solve_second_order,
};
pub use semilinear::{SemilinearOdeProblem, solve_exponential};
pub use solution::{Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use split_euler::{SplitOdeAlgorithm, solve_split_euler};
pub use symplectic::{
    SymplecticAlgorithm, SymplecticSolution, SymplecticSolveError, SymplecticTableau,
    solve_symplectic,
};
