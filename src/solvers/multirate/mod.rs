//! Multirate infinitesimal-step and MRI-GARK algorithms.

pub mod general;

pub use general::*;

pub mod prelude {
    pub use super::general::*;
}
