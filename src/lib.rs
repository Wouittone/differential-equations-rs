//! Experimental Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! This crate is a proof of concept. Its API is expected to change while
//! numerical compliance, performance, and memory behavior are established.

#![forbid(unsafe_code)]

mod adams;
mod explicit_rk;
mod implicit;
mod problem;
mod rosenbrock;
mod solution;
mod solver;
mod tsit5;

pub use adams::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54};
pub use explicit_rk::{
    Alshina2, Alshina3, Bs3, ButcherTableau, Dp5, Euler, ExplicitRungeKutta, Heun, Midpoint,
    Ralston, Ralston4, Rk4, Rkm, SspRk22, SspRk33, SspRk43,
};
pub use implicit::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
pub use problem::OdeProblem;
pub use rosenbrock::Rosenbrock23;
pub use solution::{Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use tsit5::Tsit5;
