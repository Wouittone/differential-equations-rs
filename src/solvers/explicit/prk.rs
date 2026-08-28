//! Parallel Runge--Kutta methods.
//!
//! The Rust driver evaluates stages serially today, but the tableau is the
//! exact pinned OrdinaryDiffEqPRK method. Its last two independent stages can
//! be evaluated concurrently by a future parallel stage executor.

use super::general::ResourceExplicitRungeKutta;
use differential_equations_tableau_macros::define_explicit_rk_tableau_from_file;

define_explicit_rk_tableau_from_file!(
    pub(super) KUTTA_PRK2P5_TABLEAU,
    "KuttaPRK2p5",
    "src/tableau/resources/explicit/kutta_prk2p5.json",
    crate = crate
);

/// Kutta's fifth-order parallel Runge--Kutta algorithm.
pub type KuttaPRK2p5 = ResourceExplicitRungeKutta;

#[allow(non_snake_case)]
/// Creates Kutta's `KuttaPRK2p5` algorithm using its pinned tableau.
///
/// The function spelling matches the corresponding SciML constructor.
pub const fn KuttaPRK2p5() -> KuttaPRK2p5 {
    ResourceExplicitRungeKutta::new(&KUTTA_PRK2P5_TABLEAU)
}
