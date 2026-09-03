//! Rosenbrock, Rosenbrock--W, and approximate-factorization algorithms.

/// Approximate-matrix-factorization Rosenbrock support.
pub mod amf;
/// Core Rosenbrock23 method and kernel.
pub mod general;
mod rosenbrock_dense;
pub mod rosenbrock_extended;
mod tableaux;

pub use amf::{
    AMF, AMFOperator, AmfFunction, AmfOperator, AmfProblem, build_amf_function, solve_amf,
};
pub use general::Rosenbrock23;
pub use rosenbrock_extended::*;
