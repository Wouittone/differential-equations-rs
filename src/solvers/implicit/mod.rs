//! Implicit Runge--Kutta and collocation algorithms.

pub mod firk;
pub mod general;
pub mod pdirk;
pub mod sdirk;
pub mod sdirk_cash4;

pub use firk::{AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9};
pub use general::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
pub use pdirk::{PDIRK44, Pdirk44};
pub use sdirk::*;
pub use sdirk_cash4::Cash4;

/// Basic implicit one-step algorithms.
pub mod basic {
    pub use super::general::*;
}

/// Historical grouping for singly diagonally implicit methods.
pub mod diagonally_implicit {
    pub use super::pdirk::*;
    pub use super::sdirk::*;
    pub use super::sdirk_cash4::*;
}

/// Historical grouping for fully implicit collocation methods.
pub mod fully_implicit {
    pub use super::firk::*;
}

pub mod prelude {
    pub use super::firk::*;
    pub use super::general::*;
    pub use super::pdirk::*;
    pub use super::sdirk::*;
    pub use super::sdirk_cash4::*;
}
