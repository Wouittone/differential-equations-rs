//! Numerical algorithms organized by solver family.
//!
//! Implementation modules live below their numerical family so imports make
//! ownership explicit, for example `solvers::implicit::sdirk::Sdirk2`.

pub mod automatic;
pub mod explicit;
pub mod exponential;
pub mod extrapolation;
pub mod implicit;
pub mod linear;
pub mod multirate;
pub mod multistep;
pub mod rosenbrock;
pub mod second_order;
pub mod stabilized;
pub mod taylor;

/// Compatibility namespace for the historical top-level SIMD family.
pub mod simd {
    pub use super::explicit::simd_rk::*;
}

/// Compatibility namespace for interaction-picture algorithms.
pub mod interaction_picture {
    pub use super::exponential::rkip::*;
}

/// Compatibility namespace for approximate-matrix-factorization methods.
pub mod amf {
    pub use super::rosenbrock::amf::*;
}

pub use automatic::prelude::*;
pub use explicit::prelude::*;
pub use exponential::prelude::*;
pub use extrapolation::prelude::*;
pub use implicit::prelude::*;
pub use linear::prelude::*;
pub use multirate::prelude::*;
pub use multistep::prelude::*;
pub use rosenbrock::prelude::*;
pub use second_order::prelude::*;
pub use stabilized::prelude::*;
pub use taylor::prelude::*;
