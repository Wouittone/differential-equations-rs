#[cfg(test)]
use super::TableauErrorKind;
use super::{
    ErrorEstimatorKind, FittedWeight, LazyDenseStage, RungeKuttaKind, RungeKuttaTableau,
    TableauError,
};
use serde::Deserialize;

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
pub(crate) enum Scalar {
    Integer(i64),
    Float(f64),
    Text(String),
}

impl Scalar {
    pub(crate) fn materialize(&self) -> Result<f64, TableauError> {
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

pub(super) fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value.mul_add(x, *coefficient))
}

pub(crate) fn materialize_vector(values: &[Scalar], label: &str) -> Result<Vec<f64>, TableauError> {
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

pub(crate) fn materialize_matrix(
    rows: &[Vec<Scalar>],
    label: &str,
) -> Result<Vec<Vec<f64>>, TableauError> {
    rows.iter()
        .enumerate()
        .map(|(row, values)| materialize_vector(values, &format!("{label}[{row}]")))
        .collect()
}

pub(crate) fn approximately_equal(left: f64, right: f64) -> bool {
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
mod tests;
