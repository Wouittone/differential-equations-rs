//! Runge–Kutta–Nyström solver implementations.

#[path = "rkn/adaptive.rs"]
mod adaptive;
#[path = "rkn/fixed.rs"]
mod fixed;
#[path = "rkn/irkn.rs"]
mod irkn;

pub(crate) use adaptive::solve_rkn_adaptive;
pub(crate) use fixed::solve_rkn_fixed;
pub(crate) use irkn::solve_irkn;
