//! Exponential Runge--Kutta and interaction-picture algorithms.

pub mod general;
pub mod rkip;

pub use general::*;
pub use rkip::{InteractionPictureAlgorithm, RKIP, RkipCacheStats, solve_rkip};

pub mod prelude {
    pub use super::general::*;
    pub use super::rkip::RKIP;
}
