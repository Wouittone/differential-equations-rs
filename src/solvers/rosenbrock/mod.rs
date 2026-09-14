//! Rosenbrock, Rosenbrock--W, and approximate-factorization algorithms.

/// Approximate-matrix-factorization Rosenbrock support.
pub mod amf;
mod general;
pub mod rosenbrock_extended;
mod tableaux;

pub use amf::{
    AMF, AMFOperator, AmfFunction, AmfOperator, AmfProblem, build_amf_function, solve_amf,
};
pub use general::Rosenbrock23;
pub use rosenbrock_extended::*;
pub(crate) use rosenbrock_extended::{ExtendedRosenbrockKernel, ExtendedRosenbrockMethod};
