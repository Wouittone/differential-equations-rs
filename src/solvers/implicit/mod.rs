//! Implicit Runge--Kutta and collocation algorithms.

/// Fully implicit Runge--Kutta and collocation methods.
pub mod firk;
/// Basic implicit Euler, midpoint, and trapezoid methods.
pub mod general;
/// Parallel diagonally implicit Runge--Kutta methods.
pub mod pdirk;
/// Singly diagonally implicit and additive Runge--Kutta methods.
pub mod sdirk;
/// Cash's embedded fourth-order SDIRK method.
pub mod sdirk_cash4;

pub use firk::{AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9};
pub use general::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
pub use pdirk::{PDIRK44, Pdirk44};
pub use sdirk::*;
pub use sdirk_cash4::Cash4;
