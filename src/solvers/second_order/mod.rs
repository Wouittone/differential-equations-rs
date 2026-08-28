//! Runge--Kutta--Nyström, structural, and symplectic algorithms.

/// Second-order problem, solution, structural, and RKN algorithms.
pub mod general;
pub mod symplectic;

mod coefficient_data {
    #![allow(clippy::excessive_precision)]

    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(pub(super), "coefficients/second_order/irkn.toml", crate = crate);
    define_coefficients_from_file!(
        pub(super),
        "coefficients/second_order/adaptive_rkn.toml",
        crate = crate
    );
    define_coefficients_from_file!(
        pub(super),
        "coefficients/second_order/general.toml",
        crate = crate
    );
}

pub use general::*;
pub use symplectic::*;
