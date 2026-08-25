//! Runge--Kutta--Nyström, structural, and symplectic algorithms.

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
}

pub use general::*;
pub use symplectic::*;

/// Runge--Kutta--Nyström algorithms.
pub mod rkn {
    pub use super::general::{
        DPRKN4, DPRKN5, DPRKN6, DPRKN6FM, DPRKN8, DPRKN12, Dprkn4, Dprkn5, Dprkn6, Dprkn6Fm,
        Dprkn8, Dprkn12, ERKN4, ERKN5, ERKN7, Erkn4, Erkn5, Erkn7, FineRKN4, FineRKN5, FineRkn4,
        FineRkn5, IRKN3, IRKN4, Irkn3, Irkn4, Nystrom4, Nystrom4VelocityIndependent,
        Nystrom5VelocityIndependent, Rkn4,
    };
}

/// Implicit structural-dynamics algorithms.
pub mod structural {
    pub use super::general::{GeneralizedAlpha, NewmarkBeta};
}

pub mod prelude {
    pub use super::general::*;
    pub use super::symplectic::*;
}
