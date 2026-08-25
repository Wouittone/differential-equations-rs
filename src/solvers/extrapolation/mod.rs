//! Explicit and linearly implicit extrapolation algorithms.

pub mod general;

pub use general::*;

pub mod prelude {
    pub use super::general::*;
}
