//! Experimental Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! This crate is a proof of concept. Its API is expected to change while
//! numerical compliance, performance, and memory behavior are established.

#![forbid(unsafe_code)]

mod problem;
mod solution;
mod solver;

pub use problem::OdeProblem;
pub use solution::{Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
