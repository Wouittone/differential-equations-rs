//! Explicit stabilized and implicit RKC algorithms.

pub mod general;
pub mod irkc;

mod coefficient_data {
    #![allow(clippy::excessive_precision)]

    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(
        pub(super),
        "coefficients/stabilized/methods.toml",
        crate = crate
    );
}

pub use general::*;
pub use irkc::{IRKC, solve_irkc};

pub mod prelude {
    pub use super::general::*;
    pub use super::irkc::IRKC;
}
