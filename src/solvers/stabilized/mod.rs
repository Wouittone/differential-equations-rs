//! Explicit stabilized and implicit RKC algorithms.

pub mod general;
pub mod irkc;
mod stabilized_coefficients;

pub use general::*;
pub use irkc::{IRKC, solve_irkc};

pub mod prelude {
    pub use super::general::*;
    pub use super::irkc::IRKC;
}
