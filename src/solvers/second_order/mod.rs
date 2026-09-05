//! Runge--Kutta--Nyström, structural, and symplectic algorithms.

pub mod function;
/// Second-order problem, solution, structural, and RKN algorithms.
pub mod general;
pub mod symplectic;

pub use function::SecondOrderFunction;
pub use general::*;
pub use symplectic::*;
