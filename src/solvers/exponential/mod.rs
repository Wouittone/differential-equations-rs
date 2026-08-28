//! Exponential Runge--Kutta and interaction-picture algorithms.

/// Exponential Runge--Kutta algorithms for ordinary ODE problems.
pub mod general;
/// Recycled Krylov interaction-picture solver for semilinear problems.
pub mod rkip;

pub use crate::semilinear::solve_exponential;
pub use general::*;
pub use rkip::{InteractionPictureAlgorithm, RKIP, RkipCacheStats, solve_rkip};
