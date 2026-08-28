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
