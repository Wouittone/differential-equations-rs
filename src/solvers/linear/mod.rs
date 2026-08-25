//! Exact linear, Magnus, and Lie-group integration algorithms.

pub mod general;

pub use general::*;
pub use general::{
    LieGroupAlgorithm, LinearOperatorAlgorithm, solve_lie_group, solve_linear_operator,
};

pub mod prelude {
    pub use super::general::*;
}
