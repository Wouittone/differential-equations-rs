//! Lazily parsed, resource-backed solver tableaus.
//!
//! Tableau resources are JSON documents embedded with [`include_str!`]. The
//! procedural macro validates each document with the same parser used here
//! while compiling, then the selected tableau is materialized only on first
//! use through [`LazyTableau`].

use std::sync::LazyLock;

#[doc(inline)]
pub use differential_equations_tableau_core::{
    FittedWeight, LazyDenseStage as ParsedLazyDenseStage, RungeKuttaKind, RungeKuttaTableau,
    TableauError, parse_tableau,
};

/// A lazily initialized, validated Runge--Kutta tableau.
///
/// Parse errors remain values instead of poisoning the process with a panic.
pub type LazyTableau = LazyLock<Result<RungeKuttaTableau, TableauError>>;

/// Returns a parsed lazy tableau, preserving any validation error.
pub fn load_tableau(
    resource: &'static LazyTableau,
) -> Result<&'static RungeKuttaTableau, TableauError> {
    match &**resource {
        Ok(tableau) => Ok(tableau),
        Err(error) => Err(error.clone()),
    }
}

// Transitional reexports while all legacy marker tableaus are migrated to
// resource-backed solver values.
#[doc(inline)]
pub use crate::solvers::explicit::general::{
    ButcherTableau, ExplicitRK, ExplicitRungeKutta, LazyDenseStage, ResourceExplicitRungeKutta,
};
#[doc(inline)]
pub use differential_equations_tableau_macros::define_explicit_rk_from_file;
