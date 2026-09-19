//! Fully implicit collocation Runge--Kutta methods.
//!
//! Tableaus are generated from the Radau-right and Gauss--Legendre nodes in
//! the same way as the pinned OrdinaryDiffEqFIRK implementation. The stage
//! equations are solved as one coupled dense Newton system.

mod algorithms;
mod dense;
mod kernel;
mod tableau;

#[cfg(test)]
mod tests;

pub use algorithms::*;
