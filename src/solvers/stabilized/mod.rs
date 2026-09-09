//! Explicit stabilized and implicit RKC algorithms.

pub mod general;
/// Implicit Runge--Kutta--Chebyshev solver for split problems.
pub mod irkc;
mod resources;

pub use general::*;
pub use irkc::{IRKC, solve_irkc};
