//! Parallel Runge--Kutta methods.
//!
//! The Rust driver evaluates stages serially today, but the tableau is the
//! exact pinned OrdinaryDiffEqPRK method. Its last two independent stages can
//! be evaluated concurrently by a future parallel stage executor.

use super::general::{ButcherTableau, ExplicitRungeKutta};

const EMPTY: &[f64] = &[];
const A2: &[f64] = &[1.0 / 3.0];
const A3: &[f64] = &[4.0 / 25.0, 6.0 / 25.0];
const A4: &[f64] = &[1.0 / 4.0, -3.0, 15.0 / 4.0];
const A5: &[f64] = &[6.0 / 81.0, 90.0 / 81.0, -50.0 / 81.0, 8.0 / 81.0];
const A6: &[f64] = &[6.0 / 75.0, 36.0 / 75.0, 10.0 / 75.0, 8.0 / 75.0, 0.0];
const A: &[&[f64]] = &[EMPTY, A2, A3, A4, A5, A6];
const B: &[f64] = &[
    23.0 / 192.0,
    0.0,
    125.0 / 192.0,
    0.0,
    -81.0 / 192.0,
    125.0 / 192.0,
];
const C: &[f64] = &[0.0, 1.0 / 3.0, 2.0 / 5.0, 1.0, 2.0 / 3.0, 4.0 / 5.0];

/// Kutta's six-stage, fifth-order explicit method optimized for two processors.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KuttaPrk2p5Tableau;

impl ButcherTableau for KuttaPrk2p5Tableau {
    const NODES: &'static [f64] = C;
    const COEFFICIENTS: &'static [&'static [f64]] = A;
    const WEIGHTS: &'static [f64] = B;
    const ERROR_WEIGHTS: Option<&'static [f64]> = None;
    const ORDER: usize = 5;
    const FSAL: bool = false;
}

/// Kutta's fifth-order parallel Runge--Kutta algorithm.
pub type KuttaPRK2p5 = ExplicitRungeKutta<KuttaPrk2p5Tableau>;

#[allow(non_snake_case)]
pub const fn KuttaPRK2p5() -> KuttaPRK2p5 {
    ExplicitRungeKutta::new()
}
