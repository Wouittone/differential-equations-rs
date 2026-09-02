//! Lazily parsed, resource-backed solver tableaus.
//!
//! Tableau resources are JSON documents embedded with [`include_str!`]. The
//! procedural macro validates each document with the same parser used here
//! while compiling, then the selected tableau is materialized only on first
//! use through [`LazyTableau`].

use std::sync::LazyLock;

#[doc(inline)]
pub use differential_equations_tableau_core::{
    FittedWeight, LazyDenseStage as ParsedLazyDenseStage, LinearMultistepTableau, RungeKuttaKind,
    RungeKuttaTableau, SymplecticTableau, TableauError, parse_multistep_tableau,
    parse_symplectic_tableau, parse_tableau,
};

/// A lazily initialized, validated Runge--Kutta tableau.
///
/// Parse errors remain values instead of poisoning the process with a panic.
pub type LazyTableau = LazyLock<Result<RungeKuttaTableau, TableauError>>;

/// A lazily initialized, validated drift/kick composition tableau.
pub type LazySymplecticTableau = LazyLock<Result<SymplecticTableau, TableauError>>;

/// A lazily initialized, validated constant-step linear multistep formula.
pub type LazyMultistepTableau = LazyLock<Result<LinearMultistepTableau, TableauError>>;

/// Returns a parsed lazy tableau, preserving any validation error.
pub fn load_tableau<T>(
    resource: &'static LazyLock<Result<T, TableauError>>,
) -> Result<&'static T, TableauError> {
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
#[doc(inline)]
pub use differential_equations_tableau_macros::define_multistep_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_symplectic_from_file;
