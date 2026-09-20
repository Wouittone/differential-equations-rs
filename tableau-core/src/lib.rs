//! Shared JSON parser and validator for canonical solver tableaus.

mod symplectic;
pub use symplectic::{SymplecticTableau, parse_symplectic_tableau};
mod multistep;
pub use multistep::{LinearMultistepTableau, parse_multistep_tableau};
mod variable_multistep;
pub use variable_multistep::{VariableMultistepTableau, parse_variable_multistep_tableau};
mod mri;
pub use mri::{MisTableau, MriTableau, parse_mis_tableau, parse_mri_tableau};
mod rosenbrock;
pub use rosenbrock::{RosenbrockKind, RosenbrockTableau, parse_rosenbrock_tableau};
mod rosenbrock_pair;
pub use rosenbrock_pair::{RosenbrockPairTableau, parse_rosenbrock_pair_tableau};
mod second_order;
pub use second_order::{
    IrknBootstrapSeed, IrknTableau, RknCoefficients, RungeKuttaNystromKind,
    RungeKuttaNystromTableau, parse_irkn_tableau, parse_rkn_tableau,
};
mod low_storage;
pub use low_storage::{
    AlternatingTwoNTableau, LowStorageAbcTableau, LowStorageAdaptiveController,
    LowStorageEmbeddedTableau, LowStorageEndpointEvaluation, LowStorageNodePolicy,
    LowStoragePidController, LowStorageRungeKuttaLayout, LowStorageRungeKuttaTableau,
    RegisterPipelineTableau, ThreeSTableau, parse_low_storage_tableau,
};
mod stabilized;
pub use stabilized::{
    EserkTableau, Rock2Tableau, Rock4Tableau, RockRecurrence, RockRecurrenceStage, Serk2Tableau,
    parse_eserk_tableau, parse_rock2_tableau, parse_rock4_tableau, parse_serk2_tableau,
};

mod runge_kutta;
use runge_kutta::evaluate_polynomial;
pub use runge_kutta::{RungeKuttaCoefficients, parse_numeric_expression, parse_tableau};
pub(crate) use runge_kutta::{Scalar, approximately_equal, materialize_matrix, materialize_vector};
use serde::Deserialize;
use std::{error::Error, fmt, sync::Arc};

/// Whether a canonical Runge--Kutta tableau is explicit or implicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RungeKuttaKind {
    /// A strictly lower-triangular stage matrix.
    Explicit,
    /// A stage matrix that may contain diagonal or upper-triangular entries.
    Implicit,
}

/// Semantics of the stage weights used to estimate local error.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorEstimatorKind {
    /// Difference between the primary update and an embedded companion.
    #[default]
    EmbeddedDifference,
    /// A method-specific residual estimate that is not an embedded update.
    DirectResidual,
}

/// One sparse stage evaluated only when a continuous extension is requested.
#[derive(Clone, Debug, PartialEq)]
pub struct LazyDenseStage {
    node: f64,
    coefficients: Vec<(usize, f64)>,
}

/// One primary weight represented by a rational polynomial in a runtime fit variable.
///
/// Coefficients are stored in ascending powers. The value is
/// `numerator(x) / denominator(x)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedWeight {
    stage: usize,
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl FittedWeight {
    /// Returns the zero-based stage whose primary weight is fitted.
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// Evaluates the fitted weight, returning `None` at a pole or on overflow.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        let numerator = evaluate_polynomial(&self.numerator, x);
        let denominator = evaluate_polynomial(&self.denominator, x);
        let value = numerator / denominator;
        (denominator != 0.0 && value.is_finite()).then_some(value)
    }
}

impl LazyDenseStage {
    /// Returns the stage node within the step.
    pub fn node(&self) -> f64 {
        self.node
    }

    /// Returns `(prior_stage_index, weight)` pairs for this sparse stage.
    pub fn coefficients(&self) -> &[(usize, f64)] {
        &self.coefficients
    }
}

/// A validated canonical Runge--Kutta tableau.
#[derive(Clone, Debug, PartialEq)]
pub struct RungeKuttaTableau {
    name: String,
    description: String,
    kind: RungeKuttaKind,
    order: usize,
    embedded_order: Option<usize>,
    real_stability_radius: Option<f64>,
    fsal: bool,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    c: Vec<f64>,
    error_estimator: ErrorEstimatorKind,
    error: Option<Vec<f64>>,
    second_error: Option<Vec<f64>>,
    dense: Option<Vec<Vec<f64>>>,
    lazy_dense_stages: Vec<LazyDenseStage>,
    fitted_weights: Vec<FittedWeight>,
    stage_predictors: Vec<Vec<f64>>,
}

impl RungeKuttaTableau {
    /// Returns the resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the human-readable method description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns whether the method is explicit or implicit.
    pub fn kind(&self) -> RungeKuttaKind {
        self.kind
    }

    /// Returns the classical order of the primary method.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Returns the formal order associated with the local error estimate.
    ///
    /// This is the companion order for an embedded difference, or the
    /// published estimator order for a direct residual. Implicit methods may
    /// use a higher-order companion for error estimation.
    pub fn embedded_order(&self) -> Option<usize> {
        self.embedded_order
    }

    /// Returns the extent of the primary method's stability region along the
    /// negative real axis, when supplied by the resource.
    pub fn real_stability_radius(&self) -> Option<f64> {
        self.real_stability_radius
    }

    /// Returns whether the method has the first-same-as-last property.
    pub fn fsal(&self) -> bool {
        self.fsal
    }

    /// Returns the number of stages in the method.
    pub fn stages(&self) -> usize {
        self.b.len()
    }

    /// Returns the full square Butcher stage matrix `A`.
    pub fn a(&self) -> &[Vec<f64>] {
        &self.a
    }

    /// Returns one full row of the Butcher stage matrix `A`.
    ///
    /// Returns `None` when `stage` is outside [`Self::stages`]. For an
    /// explicit method, prefer [`Self::stage_row`] when only the nonzero
    /// strictly lower-triangular prefix is needed.
    pub fn a_row(&self, stage: usize) -> Option<&[f64]> {
        self.a.get(stage).map(Vec::as_slice)
    }

    /// Returns the strictly lower-triangular prefix of an explicit stage row.
    ///
    /// Returns `None` for implicit tableaus and when `stage` is outside
    /// [`Self::stages`]. Stage zero is represented by an empty slice.
    pub fn stage_row(&self, stage: usize) -> Option<&[f64]> {
        if self.kind != RungeKuttaKind::Explicit {
            return None;
        }
        self.a.get(stage).map(|row| &row[..stage])
    }

    /// Returns the primary weights `b`.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Returns the stage nodes `c`.
    pub fn c(&self) -> &[f64] {
        &self.c
    }

    /// Returns how the error-weight vectors must be interpreted.
    pub fn error_estimator_kind(&self) -> ErrorEstimatorKind {
        self.error_estimator
    }

    /// Returns direct stage-combination error weights.
    ///
    /// Resources can provide `error` directly or `b_hat`, in which case these
    /// weights are materialized as `b - b_hat` once during parsing.
    pub fn error(&self) -> Option<&[f64]> {
        self.error.as_deref()
    }

    /// Returns a second direct error estimator, when present.
    pub fn second_error(&self) -> Option<&[f64]> {
        self.second_error.as_deref()
    }

    /// Returns continuous-extension coefficient rows.
    pub fn dense(&self) -> Option<&[Vec<f64>]> {
        self.dense.as_deref()
    }

    /// Returns stages used only by the continuous extension.
    pub fn lazy_dense_stages(&self) -> &[LazyDenseStage] {
        &self.lazy_dense_stages
    }

    /// Returns runtime-fitted primary weights for parametric RK methods.
    pub fn fitted_weights(&self) -> &[FittedWeight] {
        &self.fitted_weights
    }

    /// Returns the runtime-fitted primary weight for `stage`, when defined.
    pub fn fitted_weight(&self, stage: usize) -> Option<&FittedWeight> {
        self.fitted_weights
            .iter()
            .find(|weight| weight.stage == stage)
    }

    /// Returns initial-guess weights on prior stage derivatives for an implicit stage.
    ///
    /// Entry `j` multiplies prior derivative `k[j]` (or `h*k[j]` when solving
    /// for scaled increments). Missing or empty rows select the driver's
    /// default predictor. An out-of-range stage returns `None`.
    pub fn stage_predictor(&self, stage: usize) -> Option<&[f64]> {
        self.stage_predictors
            .get(stage)
            .filter(|row| !row.is_empty())
            .map(Vec::as_slice)
    }
}

/// Category of a tableau resource failure.
///
/// This classification is stable enough for callers to branch on without
/// parsing a human-readable diagnostic. The associated [`TableauError`]
/// retains the complete message and, when available, the originating parser
/// error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TableauErrorKind {
    /// The resource is not valid JSON or does not match its JSON shape.
    JsonSyntax,
    /// The resource declares a different method name than the requested one.
    NameMismatch,
    /// A string coefficient is not a supported numeric expression.
    NumericExpression,
    /// A parsed or evaluated coefficient is NaN or infinite.
    NonFiniteCoefficient,
    /// The decoded tableau violates a mathematical or structural invariant.
    Validation,
}

/// A failure to parse or validate a tableau resource.
#[derive(Clone, Debug)]
pub struct TableauError {
    kind: TableauErrorKind,
    message: String,
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

impl TableauError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            kind: TableauErrorKind::Validation,
            message: message.into(),
            source: None,
        }
    }

    fn json(context: &str, error: serde_json::Error) -> Self {
        Self::with_source(
            TableauErrorKind::JsonSyntax,
            format!("invalid {context} JSON: {error}"),
            error,
        )
    }

    fn name_mismatch(message: impl Into<String>) -> Self {
        Self {
            kind: TableauErrorKind::NameMismatch,
            message: message.into(),
            source: None,
        }
    }

    fn numeric_expression(message: impl Into<String>) -> Self {
        Self {
            kind: TableauErrorKind::NumericExpression,
            message: message.into(),
            source: None,
        }
    }

    fn numeric_expression_source<E>(message: impl Into<String>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::with_source(TableauErrorKind::NumericExpression, message, source)
    }

    fn non_finite(message: impl Into<String>) -> Self {
        Self {
            kind: TableauErrorKind::NonFiniteCoefficient,
            message: message.into(),
            source: None,
        }
    }

    fn with_source<E>(kind: TableauErrorKind, message: impl Into<String>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self {
            kind,
            message: message.into(),
            source: Some(Arc::new(source)),
        }
    }

    fn with_context(mut self, context: impl fmt::Display) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }

    /// Returns the machine-readable failure category.
    pub fn kind(&self) -> TableauErrorKind {
        self.kind
    }
}

impl PartialEq for TableauError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.message == other.message
    }
}

impl Eq for TableauError {}

impl fmt::Display for TableauError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TableauError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}
