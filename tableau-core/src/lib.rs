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
    IrknBootstrapSeed, IrknTableau, RungeKuttaNystromKind, RungeKuttaNystromTableau,
    parse_irkn_tableau, parse_rkn_tableau,
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

    /// Returns the full square Butcher stage matrix `A`.
    pub fn a(&self) -> &[Vec<f64>] {
        &self.a
    }

    /// Returns one strictly lower stage row of an explicit `A` matrix.
    pub fn stage_row(&self, stage: usize) -> &[f64] {
        &self.a[stage][..stage]
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

/// Parses and validates a canonical JSON tableau resource.
pub fn parse_tableau(
    source: &str,
    requested_name: &str,
) -> Result<RungeKuttaTableau, TableauError> {
    let raw: RawTableau =
        serde_json::from_str(source).map_err(|error| TableauError::json("tableau", error))?;
    if raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    raw.materialize()
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawKind {
    ExplicitRungeKutta,
    ImplicitRungeKutta,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableau {
    /// Editor-only JSON Schema hint, deliberately discarded after parsing.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    kind: RawKind,
    order: usize,
    embedded_order: Option<usize>,
    real_stability_radius: Option<Scalar>,
    #[serde(default)]
    fsal: bool,
    #[serde(rename = "A")]
    a: Vec<Vec<Scalar>>,
    b: Vec<Scalar>,
    c: Vec<Scalar>,
    b_hat: Option<Vec<Scalar>>,
    #[serde(default)]
    error_estimator: ErrorEstimatorKind,
    error: Option<Vec<Scalar>>,
    second_error: Option<Vec<Scalar>>,
    dense: Option<Vec<Vec<Scalar>>>,
    #[serde(default)]
    lazy_dense_stages: Vec<RawLazyDenseStage>,
    #[serde(default)]
    fitted_weights: Vec<RawFittedWeight>,
    #[serde(default)]
    stage_predictors: Vec<Vec<Scalar>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFittedWeight {
    stage: usize,
    numerator: Vec<Scalar>,
    denominator: Vec<Scalar>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLazyDenseStage {
    c: Scalar,
    #[serde(rename = "A")]
    a: Vec<RawSparseCoefficient>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSparseCoefficient {
    stage: usize,
    value: Scalar,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Scalar {
    Integer(i64),
    Float(f64),
    Text(String),
}

impl Scalar {
    fn materialize(&self) -> Result<f64, TableauError> {
        let value = match self {
            Self::Integer(value) => *value as f64,
            Self::Float(value) => *value,
            Self::Text(value) => parse_numeric_expression(value)?,
        };
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| TableauError::non_finite("tableau coefficients must be finite"))
    }
}

impl RawTableau {
    fn materialize(self) -> Result<RungeKuttaTableau, TableauError> {
        if self.description.trim().is_empty() {
            return Err(TableauError::new("tableau description must not be empty"));
        }
        if self.order == 0 {
            return Err(TableauError::new("tableau order must be positive"));
        }
        if self.embedded_order.is_some_and(|order| {
            order == 0 || (matches!(self.kind, RawKind::ExplicitRungeKutta) && order >= self.order)
        }) {
            return Err(TableauError::new(
                "embedded order must be positive; explicit methods require a lower-order companion",
            ));
        }
        let real_stability_radius = self
            .real_stability_radius
            .as_ref()
            .map(Scalar::materialize)
            .transpose()?;
        if real_stability_radius.is_some_and(|radius| radius <= 0.0) {
            return Err(TableauError::new(
                "real_stability_radius must be positive and finite",
            ));
        }
        if real_stability_radius.is_some() && !matches!(self.kind, RawKind::ExplicitRungeKutta) {
            return Err(TableauError::new(
                "real_stability_radius is only supported for explicit Runge--Kutta tableaus",
            ));
        }

        let a = materialize_matrix(&self.a, "A")?;
        let b = materialize_vector(&self.b, "b")?;
        let c = materialize_vector(&self.c, "c")?;
        let stages = b.len();
        if stages == 0 || c.len() != stages || a.len() != stages {
            return Err(TableauError::new(
                "A, b, and c must have the same non-zero stage count",
            ));
        }
        if a.iter().any(|row| row.len() != stages) {
            return Err(TableauError::new("A must be a square stage matrix"));
        }

        let kind = match self.kind {
            RawKind::ExplicitRungeKutta => RungeKuttaKind::Explicit,
            RawKind::ImplicitRungeKutta => RungeKuttaKind::Implicit,
        };
        if kind == RungeKuttaKind::Explicit {
            for (row, coefficients) in a.iter().enumerate() {
                if coefficients[row..].iter().any(|value| *value != 0.0) {
                    return Err(TableauError::new(format!(
                        "explicit tableau A row {row} is not strictly lower triangular"
                    )));
                }
            }
        }

        let stage_predictors = materialize_matrix(&self.stage_predictors, "stage_predictors")?;
        if !stage_predictors.is_empty() {
            if kind != RungeKuttaKind::Implicit || stage_predictors.len() != stages {
                return Err(TableauError::new(
                    "stage_predictors requires an implicit tableau and one row per stage",
                ));
            }
            for (stage, row) in stage_predictors.iter().enumerate() {
                if row.len() > stage {
                    return Err(TableauError::new(format!(
                        "stage_predictors row {stage} references an unavailable stage",
                    )));
                }
                if !row.is_empty() && !approximately_equal(row.iter().sum(), 1.0) {
                    return Err(TableauError::new(format!(
                        "stage_predictors row {stage} must sum to one",
                    )));
                }
            }
        }

        let weight_sum = b.iter().sum::<f64>();
        if !approximately_equal(weight_sum, 1.0) {
            return Err(TableauError::new(format!(
                "primary weights b must sum to one; found {weight_sum}"
            )));
        }

        if self.b_hat.is_some() && self.error_estimator != ErrorEstimatorKind::EmbeddedDifference {
            return Err(TableauError::new(
                "b_hat requires the embedded-difference error estimator",
            ));
        }
        let error = match (self.error.as_deref(), self.b_hat.as_deref()) {
            (Some(_), Some(_)) => {
                return Err(TableauError::new("provide error or b_hat, not both"));
            }
            (error, None) => materialize_optional_vector(error, "error", stages)?,
            (None, Some(weights)) => {
                if weights.len() != stages {
                    return Err(TableauError::new("b_hat must contain one entry per stage"));
                }
                let weights = materialize_vector(weights, "b_hat")?;
                let sum = weights.iter().sum::<f64>();
                if !approximately_equal(sum, 1.0) {
                    return Err(TableauError::new(format!(
                        "embedded weights b_hat must sum to one; found {sum}"
                    )));
                }
                Some(
                    b.iter()
                        .zip(weights)
                        .enumerate()
                        .map(|(stage, (b, b_hat))| {
                            let error = b - b_hat;
                            error.is_finite().then_some(error).ok_or_else(|| {
                                TableauError::non_finite(format!(
                                    "derived error[{stage}] is not finite"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };
        if error.is_none() && self.error_estimator != ErrorEstimatorKind::EmbeddedDifference {
            return Err(TableauError::new(
                "error_estimator requires error weights or b_hat",
            ));
        }
        let second_error =
            materialize_optional_vector(self.second_error.as_deref(), "second_error", stages)?;
        if second_error.is_some() && error.is_none() {
            return Err(TableauError::new("second_error requires error or b_hat"));
        }
        if self.error_estimator == ErrorEstimatorKind::EmbeddedDifference {
            for (label, weights) in [
                ("error", error.as_deref()),
                ("second_error", second_error.as_deref()),
            ] {
                let Some(weights) = weights else {
                    continue;
                };
                let sum = weights.iter().sum::<f64>();
                if !approximately_equal(sum, 0.0) {
                    return Err(TableauError::new(format!(
                        "{label} weights must sum to zero; found {sum}"
                    )));
                }
            }
        }
        if error.is_some()
            && error
                .iter()
                .chain(&second_error)
                .flat_map(|weights| weights.iter())
                .all(|weight| *weight == 0.0)
        {
            return Err(TableauError::new(
                "an error estimator must contain at least one non-zero weight",
            ));
        }
        if second_error
            .as_ref()
            .is_some_and(|weights| weights.iter().all(|weight| *weight == 0.0))
        {
            return Err(TableauError::new(
                "second_error must contain at least one non-zero weight",
            ));
        }
        if self.embedded_order.is_some() != error.is_some() {
            return Err(TableauError::new(
                "embedded_order requires exactly one of error or b_hat, and vice versa",
            ));
        }

        let lazy_dense_stages = self
            .lazy_dense_stages
            .into_iter()
            .enumerate()
            .map(|(offset, stage)| {
                let node = stage.c.materialize()?;
                let coefficients = stage
                    .a
                    .into_iter()
                    .map(|coefficient| {
                        if coefficient.stage >= stages + offset {
                            return Err(TableauError::new(
                                "lazy dense stage references an unavailable stage",
                            ));
                        }
                        Ok((coefficient.stage, coefficient.value.materialize()?))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coefficients.is_empty() {
                    return Err(TableauError::new(
                        "lazy dense stages require at least one coefficient",
                    ));
                }
                Ok(LazyDenseStage { node, coefficients })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dense = self
            .dense
            .as_deref()
            .map(|rows| materialize_matrix(rows, "dense"))
            .transpose()?;
        if let Some(rows) = &dense {
            if rows.len() != stages + lazy_dense_stages.len() || rows.iter().any(Vec::is_empty) {
                return Err(TableauError::new(
                    "dense must contain one non-empty row per ordinary and lazy stage",
                ));
            }
            for (stage, row) in rows.iter().enumerate() {
                let endpoint_weight = b.get(stage).copied().unwrap_or(0.0);
                if !approximately_equal(row.iter().sum(), endpoint_weight) {
                    return Err(TableauError::new(format!(
                        "dense row {stage} does not reproduce its endpoint weight"
                    )));
                }
            }
        } else if !lazy_dense_stages.is_empty() {
            return Err(TableauError::new("lazy dense stages require dense rows"));
        }

        let mut fitted_weights = Vec::with_capacity(self.fitted_weights.len());
        for raw_weight in self.fitted_weights {
            if raw_weight.stage >= stages {
                return Err(TableauError::new(
                    "fitted weight references an unavailable stage",
                ));
            }
            if fitted_weights
                .iter()
                .any(|weight: &FittedWeight| weight.stage == raw_weight.stage)
            {
                return Err(TableauError::new(
                    "fitted weights must reference distinct stages",
                ));
            }
            if raw_weight.numerator.is_empty() || raw_weight.denominator.is_empty() {
                return Err(TableauError::new(
                    "fitted weight polynomials must not be empty",
                ));
            }
            let numerator = materialize_vector(&raw_weight.numerator, "fitted numerator")?;
            let denominator = materialize_vector(&raw_weight.denominator, "fitted denominator")?;
            let weight = FittedWeight {
                stage: raw_weight.stage,
                numerator,
                denominator,
            };
            let zero_fit = weight.evaluate(0.0).ok_or_else(|| {
                TableauError::new("fitted weight denominator must be non-zero at zero")
            })?;
            if !approximately_equal(zero_fit, b[weight.stage]) {
                return Err(TableauError::new(format!(
                    "fitted weight for stage {} does not reproduce its zero-fit primary weight",
                    weight.stage
                )));
            }
            fitted_weights.push(weight);
        }

        if self.fsal
            && (c.first() != Some(&0.0)
                || a[0].iter().any(|value| *value != 0.0)
                || c.last() != Some(&1.0)
                || (kind == RungeKuttaKind::Explicit && b.last() != Some(&0.0))
                || a.last().is_none_or(|row| {
                    row.iter()
                        .zip(&b)
                        .any(|(stage, weight)| !approximately_equal(*stage, *weight))
                }))
        {
            return Err(TableauError::new(
                "FSAL requires an explicit first stage at c=0 and a final stage at c=1 with row b",
            ));
        }

        Ok(RungeKuttaTableau {
            name: self.name,
            description: self.description,
            kind,
            order: self.order,
            embedded_order: self.embedded_order,
            real_stability_radius,
            fsal: self.fsal,
            a,
            b,
            c,
            error_estimator: self.error_estimator,
            error,
            second_error,
            dense,
            lazy_dense_stages,
            fitted_weights,
            stage_predictors,
        })
    }
}

fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value.mul_add(x, *coefficient))
}

fn materialize_vector(values: &[Scalar], label: &str) -> Result<Vec<f64>, TableauError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .materialize()
                .map_err(|error| error.with_context(format_args!("{label}[{index}]")))
        })
        .collect()
}

fn materialize_optional_vector(
    values: Option<&[Scalar]>,
    label: &str,
    stages: usize,
) -> Result<Option<Vec<f64>>, TableauError> {
    values
        .map(|values| {
            if values.len() != stages {
                return Err(TableauError::new(format!(
                    "{label} must contain one entry per stage"
                )));
            }
            materialize_vector(values, label)
        })
        .transpose()
}

fn materialize_matrix(rows: &[Vec<Scalar>], label: &str) -> Result<Vec<Vec<f64>>, TableauError> {
    rows.iter()
        .enumerate()
        .map(|(row, values)| materialize_vector(values, &format!("{label}[{row}]")))
        .collect()
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-10 * scale
}

/// Parses the restricted arithmetic syntax accepted by numeric resource fields.
///
/// Decimal and scientific-notation literals use Rust's native `f64` parser.
/// Expressions are evaluated by `exmex` after rejecting identifiers and
/// functions other than `sqrt`, keeping every resource format on one
/// maintained expression implementation.
pub fn parse_numeric_expression(source: &str) -> Result<f64, TableauError> {
    let normalized = source.replace('_', "");
    if let Ok(value) = normalized.parse::<f64>() {
        return value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| TableauError::non_finite("coefficient expression is not finite"));
    }
    let arithmetic = normalized.replace("sqrt", "");
    if arithmetic.chars().any(|character| {
        !character.is_ascii_digit()
            && !character.is_ascii_whitespace()
            && !matches!(
                character,
                '.' | 'e' | 'E' | '+' | '-' | '*' | '/' | '(' | ')'
            )
    }) {
        return Err(TableauError::numeric_expression(format!(
            "coefficient expression `{source}` uses unsupported symbols"
        )));
    }
    let value = exmex::eval_str::<f64>(&normalized).map_err(|error| {
        TableauError::numeric_expression_source(
            format!("invalid coefficient expression `{source}`: {error}"),
            error,
        )
    })?;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| TableauError::non_finite("coefficient expression is not finite"))
}

#[cfg(test)]
mod tests {
    use super::{
        ErrorEstimatorKind, RungeKuttaKind, TableauErrorKind, parse_numeric_expression,
        parse_tableau,
    };
    use std::error::Error as _;

    const RESOURCE: &str = r#"{
      "name": "Heun",
      "description": "Heun's explicit second-order method.",
      "kind": "explicit-runge-kutta",
      "order": 2,
      "A": [[0, 0], [1, 0]],
      "b": ["1/2", "1/2"],
      "c": [0, 1]
    }"#;

    #[test]
    fn parses_canonical_butcher_tableau() {
        let tableau = parse_tableau(RESOURCE, "Heun").unwrap();
        assert_eq!(tableau.kind(), RungeKuttaKind::Explicit);
        assert_eq!(tableau.a(), &[vec![0.0, 0.0], vec![1.0, 0.0]]);
        assert_eq!(tableau.b(), &[0.5, 0.5]);
        assert_eq!(tableau.c(), &[0.0, 1.0]);
        assert_eq!(tableau.real_stability_radius(), None);
    }

    #[test]
    fn real_stability_radius_is_optional_positive_explicit_metadata() {
        let source = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"real_stability_radius\": \"4 / 2\",",
        );
        assert_eq!(
            parse_tableau(&source, "Heun")
                .unwrap()
                .real_stability_radius(),
            Some(2.0)
        );

        for invalid in [
            source.replace("\"4 / 2\"", "0"),
            source.replace("\"4 / 2\"", "-1"),
            source.replace("\"4 / 2\"", "\"1 / 0\""),
            source.replace("explicit-runge-kutta", "implicit-runge-kutta"),
        ] {
            assert!(
                parse_tableau(&invalid, "Heun").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn embedded_weights_and_direct_errors_materialize_identically() {
        let embedded = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"embedded_order\": 1, \"b_hat\": [1, 0],",
        );
        let direct = embedded.replace("\"b_hat\": [1, 0]", "\"error\": [\"-1/2\", \"1/2\"]");
        let tableau = parse_tableau(&embedded, "Heun").unwrap();
        assert_eq!(tableau, parse_tableau(&direct, "Heun").unwrap());
        assert_eq!(tableau.error(), Some([-0.5, 0.5].as_slice()));
        let second = embedded.replace(
            "\"c\": [0, 1]",
            "\"c\": [0, 1], \"second_error\": [\"-1/4\", \"1/4\"]",
        );
        assert_eq!(
            parse_tableau(&second, "Heun").unwrap().second_error(),
            Some([-0.25, 0.25].as_slice())
        );
    }

    #[test]
    fn direct_residual_estimators_are_explicitly_typed() {
        let residual = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"embedded_order\": 1, \"error_estimator\": \"direct-residual\", \"error\": [1, 0],",
        );
        let tableau = parse_tableau(&residual, "Heun").unwrap();
        assert_eq!(
            tableau.error_estimator_kind(),
            ErrorEstimatorKind::DirectResidual
        );
        assert_eq!(tableau.error(), Some([1.0, 0.0].as_slice()));

        assert!(
            parse_tableau(
                &residual.replace(", \"error_estimator\": \"direct-residual\"", ""),
                "Heun"
            )
            .is_err()
        );
        assert!(
            parse_tableau(
                &residual.replace("\"error\": [1, 0]", "\"b_hat\": [1, 0]"),
                "Heun",
            )
            .is_err()
        );
        assert!(
            parse_tableau(
                &RESOURCE.replace(
                    "\"order\": 2,",
                    "\"order\": 2, \"error_estimator\": \"direct-residual\",",
                ),
                "Heun",
            )
            .is_err()
        );
    }

    #[test]
    fn secondary_estimators_and_dense_rows_are_structurally_validated() {
        let embedded = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"embedded_order\": 1, \"error\": [\"-1/2\", \"1/2\"],",
        );
        for invalid in [
            RESOURCE.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"second_error\": [0, 0]"),
            embedded.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"second_error\": [0]"),
            embedded.replace("\"-1/2\", \"1/2\"", "0, 0"),
            embedded.replace("\"-1/2\", \"1/2\"", "1, 1"),
            RESOURCE.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"dense\": [[\"1/2\"], []]"),
            RESOURCE.replace(
                "\"c\": [0, 1]",
                "\"c\": [0, 1], \"dense\": [[1], [\"1/2\"], [0]]",
            ),
        ] {
            assert!(
                parse_tableau(&invalid, "Heun").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn implicit_companions_and_stage_predictors_are_validated() {
        // First-order endpoint update with a second-order trapezoidal companion.
        let source = r#"{"name":"Pair","description":"Implicit pair","kind":"implicit-runge-kutta","order":1,"embedded_order":2,"A":[[0,0],["1/2","1/2"]],"b":[0,1],"c":[0,1],"error":["-1/2","1/2"],"stage_predictors":[[],[1]]}"#;
        let tableau = parse_tableau(source, "Pair").unwrap();
        assert_eq!(tableau.embedded_order(), Some(2));
        assert_eq!(tableau.stage_predictor(1), Some([1.0].as_slice()));
        assert_eq!(tableau.stage_predictor(0), None);
        assert_eq!(tableau.stage_predictor(2), None);
        for invalid in [
            source.replace("[[],[1]]", "[[]]"),
            source.replace("[[],[1]]", "[[1],[1]]"),
            source.replace("[[],[1]]", "[[],[1,0]]"),
            source.replace("[[],[1]]", "[[],[2]]"),
            source.replace("[[],[1]]", "[[],[\"1/0\"]]"),
            source.replace("[[],[1]]", "null"),
            source.replace("\"embedded_order\":2", "\"embedded_order\":0"),
            source.replace("implicit-runge-kutta", "explicit-runge-kutta"),
        ] {
            assert!(
                parse_tableau(&invalid, "Pair").is_err(),
                "accepted {invalid}"
            );
        }
        let defaults = source.replace("[[],[1]]", "[[],[]]");
        assert_eq!(
            parse_tableau(&defaults, "Pair").unwrap().stage_predictor(1),
            None
        );
        let explicit = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"stage_predictors\": [[],[1]],",
        );
        assert!(parse_tableau(&explicit, "Heun").is_err());
    }

    #[test]
    fn fsal_checks_both_endpoint_stages_for_explicit_and_implicit_tableaus() {
        let implicit = r#"{"name":"Trap","description":"Trapezoidal rule","kind":"implicit-runge-kutta","order":2,"fsal":true,"A":[[0,0],["1/2","1/2"]],"b":["1/2","1/2"],"c":[0,1]}"#;
        assert!(parse_tableau(implicit, "Trap").unwrap().fsal());
        for invalid in [
            implicit.replace("[0,1]", "[0.1,1]"),
            implicit.replace("[0,1]", "[0,0.5]"),
            implicit.replace("[0,0]", "[\"1/2\",\"-1/2\"]"),
            implicit.replace("\"b\":[\"1/2\",\"1/2\"]", "\"b\":[1,0]"),
        ] {
            assert!(
                parse_tableau(&invalid, "Trap").is_err(),
                "accepted {invalid}"
            );
        }
        let explicit = r#"{"name":"Heun","description":"Heun with a final FSAL stage","kind":"explicit-runge-kutta","order":2,"fsal":true,"A":[[0,0,0],[1,0,0],["1/2","1/2",0]],"b":["1/2","1/2",0],"c":[0,1,1]}"#;
        assert!(parse_tableau(explicit, "Heun").unwrap().fsal());
    }

    #[test]
    fn rejects_invalid_or_ambiguous_embedded_weights() {
        let embedded = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"embedded_order\": 1, \"b_hat\": [1, 0],",
        );
        for invalid in [
            embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1]"),
            embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1, 1]"),
            embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [\"1/0\", 0]"),
            embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1e308, 1e308]"),
            embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1, 0], \"error\": [0, 0]"),
            embedded.replace("\"embedded_order\": 1,", ""),
            embedded.replace("\"b_hat\": [1, 0],", ""),
        ] {
            assert!(
                parse_tableau(&invalid, "Heun").is_err(),
                "accepted {invalid}"
            );
        }
        // Each weight sum is finite and one, but subtraction can still overflow.
        let overflow = r#"{"name":"Overflow","description":"Subtraction overflow","kind":"explicit-runge-kutta","order":2,"embedded_order":1,"A":[[0,0,0],[0,0,0],[0,0,0]],"b":[1e308,-1e308,1],"b_hat":[-1e308,1e308,1],"c":[0,0,0]}"#;
        let error = parse_tableau(overflow, "Overflow").unwrap_err();
        assert!(error.to_string().contains("derived error[0] is not finite"));
    }

    #[test]
    fn overflowing_coefficient_sums_are_not_approximately_consistent() {
        let weights = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1e308, 1e308]");
        assert!(parse_tableau(&weights, "Heun").is_err());
        let dense = RESOURCE.replace(
            "\"c\": [0, 1]",
            "\"c\": [0, 1], \"dense\": [[1e308, 1e308], [\"1/2\"]]",
        );
        assert!(parse_tableau(&dense, "Heun").is_err());
    }

    #[test]
    fn expression_parser_supports_exact_style_coefficients() {
        assert_eq!(
            parse_numeric_expression("(3 - sqrt(3)) / 6").unwrap(),
            (3.0 - 3.0_f64.sqrt()) / 6.0
        );
        assert_eq!(parse_numeric_expression("1_000 / 4").unwrap(), 250.0);
        assert_eq!(parse_numeric_expression("-3.25e-7").unwrap(), -3.25e-7);
    }

    #[test]
    fn expression_parser_exposes_only_the_tableau_math_context() {
        assert!(parse_numeric_expression("pi").is_err());
        assert!(parse_numeric_expression("sin(1)").is_err());
        assert!(parse_numeric_expression("coefficient + 1").is_err());
    }

    #[test]
    fn tableau_failures_expose_stable_categories_and_parser_sources() {
        let json = parse_tableau("{", "Heun").unwrap_err();
        assert_eq!(json.kind(), TableauErrorKind::JsonSyntax);
        assert!(json.source().is_some());

        let name = parse_tableau(RESOURCE, "Other").unwrap_err();
        assert_eq!(name.kind(), TableauErrorKind::NameMismatch);
        assert!(name.source().is_none());

        let expression = parse_numeric_expression("1+").unwrap_err();
        assert_eq!(expression.kind(), TableauErrorKind::NumericExpression);
        assert!(expression.source().is_some());

        let non_finite = parse_numeric_expression("1e999").unwrap_err();
        assert_eq!(non_finite.kind(), TableauErrorKind::NonFiniteCoefficient);

        let invalid = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1, 1]");
        assert_eq!(
            parse_tableau(&invalid, "Heun").unwrap_err().kind(),
            TableauErrorKind::Validation
        );
    }

    #[test]
    fn rejects_structurally_invalid_resources() {
        let nonsquare = RESOURCE.replace("[1, 0]", "[1]");
        assert!(parse_tableau(&nonsquare, "Heun").is_err());

        let nonexplicit = RESOURCE.replace("[1, 0]", "[1, 1]");
        assert!(parse_tableau(&nonexplicit, "Heun").is_err());

        let bad_weights = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1, 1]");
        assert!(parse_tableau(&bad_weights, "Heun").is_err());

        let unknown = RESOURCE.replace("\"order\": 2,", "\"order\": 2, \"mystery\": 1,");
        assert!(parse_tableau(&unknown, "Heun").is_err());
    }

    #[test]
    fn rejects_invalid_expressions_before_runtime_use() {
        let division_by_zero = RESOURCE.replace("\"1/2\"", "\"1/0\"");
        assert!(parse_tableau(&division_by_zero, "Heun").is_err());
    }

    #[test]
    fn schema_reference_is_ignored() {
        let source = RESOURCE.replace("{\n", "{\n      \"$schema\": \"../schema.json\",\n");
        assert!(parse_tableau(&source, "Heun").is_ok());
    }

    #[test]
    fn schema_reference_is_typed_without_weakening_unknown_field_checks() {
        let invalid_schema =
            RESOURCE.replace("{\n", "{\n      \"$schema\": {\"unexpected\": true},\n");
        let error = parse_tableau(&invalid_schema, "Heun").unwrap_err();
        assert!(error.to_string().contains("invalid type"));

        let unknown = RESOURCE.replace("\"order\": 2,", "\"order\": 2, \"typo\": 1,");
        let error = parse_tableau(&unknown, "Heun").unwrap_err();
        assert!(error.to_string().contains("unknown field `typo`"));
    }

    #[test]
    fn parses_and_validates_runtime_fitted_weights() {
        let fitted = RESOURCE.replace(
            "\"c\": [0, 1]",
            "\"c\": [0, 1], \"fitted_weights\": [{\"stage\": 0, \
             \"numerator\": [\"1/2\", 1], \"denominator\": [1, \"1/2\"]}]",
        );
        let tableau = parse_tableau(&fitted, "Heun").unwrap();
        let weight = tableau.fitted_weight(0).unwrap();
        assert_eq!(weight.stage(), 0);
        assert_eq!(weight.evaluate(2.0), Some(1.25));

        let wrong_zero_fit = fitted.replace("[\"1/2\", 1]", "[\"1/3\", 1]");
        assert!(parse_tableau(&wrong_zero_fit, "Heun").is_err());

        let duplicate = fitted.replace(
            "]}",
            "]}, {\"stage\": 0, \"numerator\": [\"1/2\"], \
             \"denominator\": [1]}]",
        );
        assert!(parse_tableau(&duplicate, "Heun").is_err());
    }
}
