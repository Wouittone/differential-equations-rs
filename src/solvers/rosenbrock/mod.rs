//! Rosenbrock, Rosenbrock--W, and approximate-factorization algorithms.

pub mod amf;
pub mod general;
mod rosenbrock_dense;
pub mod rosenbrock_extended;

pub use amf::{
    AMF, AMFOperator, AmfFunction, AmfOperator, AmfProblem, build_amf_function, solve_amf,
};
pub use general::Rosenbrock23;
pub use rosenbrock_extended::*;

pub mod prelude {
    pub use super::amf::AMF;
    pub use super::general::Rosenbrock23;
    pub use super::rosenbrock_extended::*;
}
