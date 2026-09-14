//! Numerical algorithms organized by solver family.
//!
//! Algorithms live below their numerical family, while shared drivers and
//! workspaces remain private. For example, use `solvers::implicit::Sdirk2`.
//!
//! ```
//! use differential_equations::solvers::{explicit::Rk4, implicit::Sdirk2};
//!
//! let _ = (Rk4, Sdirk2);
//! ```
//!
//! Implementation namespaces are intentionally inaccessible:
//!
//! ```compile_fail
//! use differential_equations::solvers::explicit::general::Rk4;
//! ```

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
