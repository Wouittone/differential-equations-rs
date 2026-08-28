//! Runge--Kutta--Nyström, structural, and symplectic algorithms.

/// Second-order problem, solution, structural, and RKN algorithms.
pub mod general;
pub mod symplectic;

mod coefficient_data {
    #![allow(clippy::excessive_precision)]

    use differential_equations_tableau_macros::define_tableau_data_from_file;

    define_tableau_data_from_file!(pub(super), "src/tableau/resources/methods/second_order/irkn.json", crate = crate);
    define_tableau_data_from_file!(
        pub(super),
        "src/tableau/resources/methods/second_order/adaptive_rkn.json",
        crate = crate
    );
    define_tableau_data_from_file!(
        pub(super),
        "src/tableau/resources/methods/second_order/general.json",
        crate = crate
    );
}

pub use general::*;
pub use symplectic::*;
