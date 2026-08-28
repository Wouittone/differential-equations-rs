//! Shared JSON parser and validator for canonical solver tableaus.

use serde::Deserialize;
use std::fmt;

/// Whether a canonical Runge--Kutta tableau is explicit or implicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RungeKuttaKind {
    /// A strictly lower-triangular stage matrix.
    Explicit,
    /// A stage matrix that may contain diagonal or upper-triangular entries.
    Implicit,
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
    fsal: bool,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    c: Vec<f64>,
    error: Option<Vec<f64>>,
    second_error: Option<Vec<f64>>,
    dense: Option<Vec<Vec<f64>>>,
    lazy_dense_stages: Vec<LazyDenseStage>,
    fitted_weights: Vec<FittedWeight>,
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

    /// Returns the lower order of an embedded pair, when present.
    pub fn embedded_order(&self) -> Option<usize> {
        self.embedded_order
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

    /// Returns direct stage-combination error weights.
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
}

/// A failure to parse or validate a tableau resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableauError {
    message: String,
}

impl TableauError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TableauError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TableauError {}

/// Parses and validates a canonical JSON tableau resource.
pub fn parse_tableau(
    source: &str,
    requested_name: &str,
) -> Result<RungeKuttaTableau, TableauError> {
    let raw: RawTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid tableau JSON: {error}")))?;
    if raw.name != requested_name {
        return Err(TableauError::new(format!(
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
    #[serde(default)]
    fsal: bool,
    #[serde(rename = "A")]
    a: Vec<Vec<Scalar>>,
    b: Vec<Scalar>,
    c: Vec<Scalar>,
    error: Option<Vec<Scalar>>,
    second_error: Option<Vec<Scalar>>,
    dense: Option<Vec<Vec<Scalar>>>,
    #[serde(default)]
    lazy_dense_stages: Vec<RawLazyDenseStage>,
    #[serde(default)]
    fitted_weights: Vec<RawFittedWeight>,
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
            .ok_or_else(|| TableauError::new("tableau coefficients must be finite"))
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
        if self
            .embedded_order
            .is_some_and(|order| order == 0 || order >= self.order)
        {
            return Err(TableauError::new(
                "embedded order must be positive and below the primary order",
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

        let weight_sum = b.iter().sum::<f64>();
        if !approximately_equal(weight_sum, 1.0) {
            return Err(TableauError::new(format!(
                "primary weights b must sum to one; found {weight_sum}"
            )));
        }

        let error = materialize_optional_vector(self.error.as_deref(), "error", stages)?;
        let second_error =
            materialize_optional_vector(self.second_error.as_deref(), "second_error", stages)?;
        if second_error.is_some() && error.is_none() {
            return Err(TableauError::new("second_error requires error"));
        }
        if self.embedded_order.is_some() != error.is_some() {
            return Err(TableauError::new(
                "embedded_order and error must either both be present or both be absent",
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
            && (kind != RungeKuttaKind::Explicit
                || c.last() != Some(&1.0)
                || b.last() != Some(&0.0)
                || a.last().is_none_or(|row| {
                    row[..stages - 1]
                        .iter()
                        .zip(&b[..stages - 1])
                        .any(|(stage, weight)| !approximately_equal(*stage, *weight))
                }))
        {
            return Err(TableauError::new(
                "FSAL requires the final explicit stage row to equal b",
            ));
        }

        Ok(RungeKuttaTableau {
            name: self.name,
            description: self.description,
            kind,
            order: self.order,
            embedded_order: self.embedded_order,
            fsal: self.fsal,
            a,
            b,
            c,
            error,
            second_error,
            dense,
            lazy_dense_stages,
            fitted_weights,
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
                .map_err(|error| TableauError::new(format!("{label}[{index}]: {error}")))
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
    (left - right).abs() <= 1.0e-10 * scale
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
            .ok_or_else(|| TableauError::new("coefficient expression is not finite"));
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
        return Err(TableauError::new(format!(
            "coefficient expression `{source}` uses unsupported symbols"
        )));
    }
    let value = exmex::eval_str::<f64>(&normalized).map_err(|error| {
        TableauError::new(format!(
            "invalid coefficient expression `{source}`: {error}"
        ))
    })?;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| TableauError::new("coefficient expression is not finite"))
}

#[cfg(test)]
mod tests {
    use super::{RungeKuttaKind, parse_numeric_expression, parse_tableau};

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
        let source = RESOURCE.replace(
            "{\n",
            "{\n      \"$schema\": \"../../docs/tableau.schema.json\",\n",
        );
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
