//! Experimental Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! This crate is a proof of concept. Its API is expected to change while
//! numerical compliance, performance, and memory behavior are established.

#![forbid(unsafe_code)]

mod abdf2;
mod adams;
mod anas5;
mod callback;
mod coefficients;
mod explicit_rk;
mod generated_coefficients;
mod implicit;
mod integrator;
mod linear;
mod low_storage_rk;
mod mebdf2;
mod problem;
mod qndf1;
mod qndf2;
mod rosenbrock;
mod rosenbrock_extended;
mod sdirk;
mod second_order;
mod solution;
mod solver;
mod ssprk_extended;
mod trbdf2;
mod tsit5;
mod variable_adams;
mod verner;

pub use abdf2::Abdf2;
pub use adams::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54};
pub use anas5::Anas5;
pub use callback::{CallbackAction, EventDirection};
pub use explicit_rk::{
    Alshina2, Alshina3, Alshina6, Bs3, Bs5, ButcherTableau, Dp5, Euler, ExplicitRungeKutta, Heun,
    Midpoint, OwrenZen3, OwrenZen4, OwrenZen5, Ralston, Ralston4, Rk4, Rkm, SspRk22, SspRk33,
    SspRk43,
};
pub use implicit::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
pub use low_storage_rk::{
    CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134, Ndblsrk144,
    Ork256, ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S82, Shlddrk64,
};
pub use mebdf2::Mebdf2;
pub use problem::{MassMatrixOdeProblem, OdeProblem, SplitOdeProblem};
pub use qndf1::Qndf1;
pub use qndf2::Qndf2;
pub use rosenbrock::Rosenbrock23;
pub use rosenbrock_extended::{Rodas4, Rodas5P, Rosenbrock32};
pub use sdirk::Sdirk2;
pub use second_order::{
    LeapfrogDriftKickDrift, SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolution,
    SecondOrderSolveError, SymplecticEuler, VelocityVerlet, VerletLeapfrog, solve_second_order,
};
pub use solution::{Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use ssprk_extended::{
    Prrk22, SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83,
    SspRk104, SspRk432, pRRK22,
};
pub use trbdf2::Trbdf2;
pub use tsit5::Tsit5;
pub use variable_adams::{Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5};
pub use verner::{Vern6, Vern7, Vern8, Vern9};
