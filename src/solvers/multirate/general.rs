//! Multirate infinitesimal-step and MRI-GARK methods for typed split ODEs.
//!
//! The slow component is the `implicit` half of [`crate::SplitOdeProblem`]; the
//! fast component is its `explicit` half. This matches
//! `OrdinaryDiffEqMultirate`'s `SplitFunction(fast, slow)` convention.

mod algorithms;
mod driver;
mod evaluation;
mod kernels;
mod method;

pub use algorithms::*;
