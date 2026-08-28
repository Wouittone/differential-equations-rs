//! Exact linear, Magnus, and Lie-group integration algorithms.

/// Dense linear-operator, Magnus, and Lie-group algorithms.
pub mod general;

pub use general::*;
pub use general::{
    LieGroupAlgorithm, LinearOperatorAlgorithm, solve_lie_group, solve_linear_operator,
};
