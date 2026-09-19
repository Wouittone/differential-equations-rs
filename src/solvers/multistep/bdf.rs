//! Variable-order regular-ODE backward differentiation formulas.
//!
//! The implementations in this module follow the identity-mass, regular-ODE
//! paths in OrdinaryDiffEqBDF at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. Residual DAEs and singular
//! mass matrices are deliberately outside the crate's current problem model.

use crate::integrator::ControllerConfig;

mod config;
mod helpers;
mod kernel;

pub use config::{FBDF, Fbdf, QBDF, QNDF, Qbdf, Qndf};

pub(super) const MAX_ORDER: usize = 5;
pub(super) const DIFFERENCE_COUNT: usize = MAX_ORDER + 2;
pub(super) const MAX_NEWTON_ITERATIONS: usize = 12;
pub(super) const NEWTON_TOLERANCE: f64 = 1.0e-12;

// A sixth-order controller exponent is converted back to each BDF order in
// `reported_error`, so the shared driver uses the pinned 1/(k+1) exponent.
pub(super) const CONTROLLER: ControllerConfig =
    ControllerConfig::proportional(6, 1.0 / 1.2, 0.1, 10.0, 0.2);
