//! Lazily parsed, resource-backed solver tableaus.
//!
//! Tableau resources are JSON documents embedded with [`include_str!`]. The
//! procedural macros validate each document with the same parsers used here
//! while compiling, then the selected tableau is materialized only on first
//! use. This includes ordinary Runge--Kutta, Runge--Kutta--Nyström, improved
//! RKN, low-storage and stabilized Runge--Kutta, multistep, Rosenbrock,
//! multirate, and symplectic representations.

use std::sync::LazyLock;
use thiserror::Error;

#[doc(inline)]
pub use differential_equations_tableau_core::{
    AlternatingTwoNTableau, ErrorEstimatorKind, EserkTableau, FittedWeight, IrknBootstrapSeed,
    IrknTableau, LazyDenseStage as ParsedLazyDenseStage, LinearMultistepTableau,
    LowStorageAbcTableau, LowStorageAdaptiveController, LowStorageEmbeddedTableau,
    LowStorageEndpointEvaluation, LowStorageNodePolicy, LowStoragePidController,
    LowStorageRungeKuttaLayout, LowStorageRungeKuttaTableau, MisTableau, MriTableau,
    RegisterPipelineTableau, Rock2Tableau, Rock4Tableau, RockRecurrence, RockRecurrenceStage,
    RosenbrockKind, RosenbrockPairTableau, RosenbrockTableau, RungeKuttaKind,
    RungeKuttaNystromKind, RungeKuttaNystromTableau, RungeKuttaTableau, Serk2Tableau,
    SymplecticTableau, TableauError, TableauErrorKind, ThreeSTableau, VariableMultistepTableau,
    parse_eserk_tableau, parse_irkn_tableau, parse_low_storage_tableau, parse_mis_tableau,
    parse_mri_tableau, parse_multistep_tableau, parse_rkn_tableau, parse_rock2_tableau,
    parse_rock4_tableau, parse_rosenbrock_pair_tableau, parse_rosenbrock_tableau,
    parse_serk2_tableau, parse_symplectic_tableau, parse_tableau, parse_variable_multistep_tableau,
};

/// A failure to select or materialize a requested tableau.
///
/// Fixed-method accessors normally expose [`TableauError`] directly. Families
/// whose public accessors can reject a runtime formula selection use this
/// error so callers can distinguish an unsupported order from a malformed
/// embedded resource. Related fixed-order accessors may use the same type to
/// keep one error contract across their family.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum TableauAccessError {
    /// The requested formula order lies outside the supported inclusive range.
    #[error(
        "tableau order {requested} is unsupported; expected an order from {minimum} through {maximum}"
    )]
    UnsupportedOrder {
        /// Requested formula order.
        requested: usize,
        /// Smallest supported order.
        minimum: usize,
        /// Largest supported order.
        maximum: usize,
    },
    /// The embedded tableau resource could not be parsed or validated.
    #[error("{0}")]
    Resource(#[from] TableauError),
    /// A parsed resource does not satisfy the selected formula family's
    /// invariants.
    #[error("tableau resource is incompatible with {context}")]
    IncompatibleFormula {
        /// Formula family or invariant that rejected the resource.
        context: &'static str,
    },
}

impl From<TableauAccessError> for crate::SolveError {
    fn from(error: TableauAccessError) -> Self {
        match error {
            TableauAccessError::Resource(error) => Self::TableauResource(error),
            TableauAccessError::UnsupportedOrder { .. }
            | TableauAccessError::IncompatibleFormula { .. } => Self::InvalidTableau,
        }
    }
}

/// A lazily initialized, validated Runge--Kutta tableau.
///
/// Parse errors remain values instead of poisoning the process with a panic.
pub type LazyTableau = LazyLock<Result<RungeKuttaTableau, TableauError>>;

/// A lazily initialized, validated drift/kick composition tableau.
pub type LazySymplecticTableau = LazyLock<Result<SymplecticTableau, TableauError>>;

/// A lazily initialized, validated constant-step linear multistep formula.
pub type LazyMultistepTableau = LazyLock<Result<LinearMultistepTableau, TableauError>>;

/// A lazily initialized, validated variable-step two-step formula.
pub type LazyVariableMultistepTableau = LazyLock<Result<VariableMultistepTableau, TableauError>>;

/// A lazily initialized, validated MRI-GARK coupling tableau.
pub type LazyMriTableau = LazyLock<Result<MriTableau, TableauError>>;

/// A lazily initialized, validated multirate infinitesimal-step tableau.
pub type LazyMisTableau = LazyLock<Result<MisTableau, TableauError>>;

/// A lazily initialized, validated Rosenbrock tableau.
pub type LazyRosenbrockTableau = LazyLock<Result<RosenbrockTableau, TableauError>>;

/// A lazily initialized, validated low-storage Rosenbrock 2/3 pair.
pub type LazyRosenbrockPairTableau = LazyLock<Result<RosenbrockPairTableau, TableauError>>;

/// A lazily initialized, validated explicit Runge--Kutta--Nyström tableau.
pub type LazyRungeKuttaNystromTableau = LazyLock<Result<RungeKuttaNystromTableau, TableauError>>;

/// A lazily initialized, validated improved RKN history tableau.
pub type LazyIrknTableau = LazyLock<Result<IrknTableau, TableauError>>;

/// A lazily initialized, validated low-storage Runge--Kutta recurrence.
pub type LazyLowStorageRungeKuttaTableau =
    LazyLock<Result<LowStorageRungeKuttaTableau, TableauError>>;

/// A lazily initialized, degree-specific ROCK2 recurrence.
pub type LazyRock2Tableau = LazyLock<Result<Rock2Tableau, TableauError>>;

/// A lazily initialized, degree-specific ROCK4 recurrence and finishing pair.
pub type LazyRock4Tableau = LazyLock<Result<Rock4Tableau, TableauError>>;

/// A lazily initialized, degree-specific SERK2 recurrence and output weights.
pub type LazySerk2Tableau = LazyLock<Result<Serk2Tableau, TableauError>>;

/// A lazily initialized, degree-specific extrapolated stabilized recurrence.
pub type LazyEserkTableau = LazyLock<Result<EserkTableau, TableauError>>;

/// Returns a parsed lazy tableau, preserving any validation error.
pub fn load_tableau<T>(
    resource: &'static LazyLock<Result<T, TableauError>>,
) -> Result<&'static T, TableauError> {
    match &**resource {
        Ok(tableau) => Ok(tableau),
        Err(error) => Err(error.clone()),
    }
}

#[doc(inline)]
pub use crate::solvers::explicit::ResourceExplicitRungeKutta;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_eserk_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_explicit_rk_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_irkn_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_low_storage_rk_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_low_storage_rk_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_mis_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_mri_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_multistep_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rkn_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rkn_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rock2_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rock4_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rosenbrock_pair_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_rosenbrock_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_serk2_tableau_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_symplectic_from_file;
#[doc(inline)]
pub use differential_equations_tableau_macros::define_variable_multistep_tableau_from_file;

#[cfg(test)]
mod tests {
    use super::{TableauAccessError, parse_tableau};
    use std::error::Error as _;

    #[test]
    fn access_errors_report_the_supported_order_range() {
        let error = TableauAccessError::UnsupportedOrder {
            requested: 6,
            minimum: 1,
            maximum: 5,
        };

        assert_eq!(
            error.to_string(),
            "tableau order 6 is unsupported; expected an order from 1 through 5"
        );
        assert!(error.source().is_none());
    }

    #[test]
    fn access_errors_preserve_resource_failures_as_sources() {
        let resource = parse_tableau("{", "Broken").unwrap_err();
        let expected = resource.to_string();
        let error = TableauAccessError::from(resource);

        assert_eq!(error.to_string(), expected);
        assert_eq!(error.source().unwrap().to_string(), expected);
    }
}
