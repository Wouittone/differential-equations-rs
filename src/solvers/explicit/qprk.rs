//! Quadruple-precision-oriented explicit Runge--Kutta methods.

use super::general::ResourceExplicitRungeKutta;
use differential_equations_tableau_macros::define_explicit_rk_tableau_from_file;

define_explicit_rk_tableau_from_file!(
    pub(super) QPRK98_TABLEAU,
    "QPRK98",
    "tableaux/explicit/qprk98.json",
    crate = crate
);

/// The QPRK98 explicit embedded 9(8) algorithm.
pub type QPRK98 = ResourceExplicitRungeKutta;

#[allow(non_snake_case)]
/// Creates the `QPRK98` embedded 9(8) algorithm using its pinned tableau.
///
/// The function spelling matches the corresponding SciML constructor.
pub const fn QPRK98() -> QPRK98 {
    ResourceExplicitRungeKutta::new(&QPRK98_TABLEAU)
}
