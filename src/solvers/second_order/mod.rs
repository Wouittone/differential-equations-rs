//! Runge--Kutta--Nyström, structural, and symplectic algorithms.

mod function;
mod general;
pub mod symplectic;

pub use function::SecondOrderFunction;
pub use general::*;
pub use symplectic::*;
